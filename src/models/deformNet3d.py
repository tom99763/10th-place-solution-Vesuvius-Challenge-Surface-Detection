# Retry with corrected UNet upsampling (uses trilinear interpolation to avoid size mismatches).
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

# Diffeo exponentiation and warper (same as before)
def make_base_grid(B, D, H, W, device):
    zz = torch.linspace(0, D-1, D, device=device)
    yy = torch.linspace(0, H-1, H, device=device)
    xx = torch.linspace(0, W-1, W, device=device)
    zz, yy, xx = torch.meshgrid(zz, yy, xx, indexing='ij')  # D,H,W
    grid = torch.stack((xx, yy, zz), dim=3)  # D,H,W,3 (x,y,z)
    grid = grid.unsqueeze(0).repeat(B,1,1,1,1)  # B,D,H,W,3
    return grid

def disp_to_grid_for_sampling(disp_voxel: torch.Tensor):
    B, C, D, H, W = disp_voxel.shape
    device = disp_voxel.device
    grid = make_base_grid(B, D, H, W, device)  # B,D,H,W,3 (x,y,z)
    disp = disp_voxel.permute(0,2,3,4,1)  # B,D,H,W,3 (dx,dy,dz)
    pos = grid + disp
    pos_norm = torch.empty_like(pos)
    pos_norm[...,0] = 2.0 * pos[...,0] / max(W-1,1) - 1.0  # x
    pos_norm[...,1] = 2.0 * pos[...,1] / max(H-1,1) - 1.0  # y
    pos_norm[...,2] = 2.0 * pos[...,2] / max(D-1,1) - 1.0  # z
    return pos_norm

def warp_vol_using_disp(vol: torch.Tensor, disp_voxel: torch.Tensor, mode='bilinear'):
    pos_norm = disp_to_grid_for_sampling(disp_voxel)
    warped = F.grid_sample(vol, pos_norm, mode=mode, padding_mode='border', align_corners=True)
    return warped

def warp_displacement(disp_voxel: torch.Tensor, by_disp_voxel: torch.Tensor):
    warped = warp_vol_using_disp(by_disp_voxel, disp_voxel, mode='bilinear')
    return warped

def scaling_and_squaring(v: torch.Tensor, n_steps:int=6) -> torch.Tensor:
    flow = v / (2.0 ** n_steps)
    for _ in range(n_steps):
        flowed = warp_displacement(flow, flow)
        flow = flow + flowed
    return flow


class DeformDynUnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictor = instantiate(cfg.models)
        self.cfg = cfg

    def forward(self, x, return_params = False):
        #x: (batch, 2, d, h, w)
        raw_v = self.predictor(x) #stationary velocity field (SVF)
        soft_oof = x[:, 1:2, :, :, :]
        v = torch.tanh(raw_v) * self.cfg.max_v
        phi = scaling_and_squaring(v, n_steps=self.cfg.n_steps)
        warped = warp_vol_using_disp(soft_oof, phi)
        if return_params:
            return warped, v, phi
        return warped

# --------------------
# --------------------
# Inverse Compositional Deformation Network 3D
# --------------------
# --------------------
class ICDeformDynUnet(DeformDynUnet):
    """
    Iterative inverse-compositional deformable warper for 3D masks.
    Predictor: shared network that predicts a small SVF (B,3,D,H,W) given [vol, warped_mask]
    At each iteration:
      - delta_v = predictor([vol, warped_mask])  (small SVF)
      - delta_phi = exp(delta_v)  (scaling-and-squaring)
      - inv_delta_phi = exp(-delta_v)  (exact inverse in SVF group)
      - phi <- inv_delta_phi + warp(phi, inv_delta_phi)  (IC composition: phi <- phi ∘ Δφ^{-1})
      - warped_mask = warp(orig_mask, phi)
    Notes:
      - We always sample the mask from the original template (deferred warping).
      - warp_displacement(field, by_disp) warps `field` by `by_disp`.
    """
    def __init__(self, cfg, predictor=None):
        super().__init__(cfg)
        # predictor already set by parent (instantiate(cfg.models))
        self.num_iters = getattr(cfg, "num_iters", 3)
        self.max_delta_v = getattr(cfg, "max_delta_v", cfg.max_v if hasattr(cfg, "max_v") else 1.0)
        self.n_steps = getattr(cfg, "n_steps", cfg.n_steps if hasattr(cfg, "n_steps") else 6)

    def forward(self, x, return_params=False):
        """
        x is concatenation of vol and mask: B, 2, D, H, W
        returns warped_mask; if return_params=True also returns list of phis and final phi
        """
        vol, mask = x[:, 0:1], x[:, 1:2]
        B, _, D, H, W = mask.shape
        device = mask.device

        # initialize phi = zero displacement (identity)
        phi = torch.zeros(B, 3, D, H, W, device=device, dtype=mask.dtype)

        orig_mask = mask
        warped_mask = orig_mask  # initial

        phis = []

        for it in range(self.num_iters):
            # predictor input: volume and current warped mask
            inp = torch.cat([vol, warped_mask], dim=1)  # B, C_img + C_mask, D,H,W
            raw_delta_v = self.predictor(inp)  # expected B,3,D,H,W

            if raw_delta_v.shape[1] != 3:
                raise RuntimeError(f"predictor must output 3 channels for voxel SVF, got {raw_delta_v.shape}")

            # small incremental SVF
            delta_v = torch.tanh(raw_delta_v) * self.max_delta_v  # keep small

            # exponentiate to delta_phi and inverse via -delta_v
            # Δφ = exp(Δv); Δφ^{-1} = exp(-Δv) exactly in SVF paramization
            # We compute only inv_delta_phi explicitly (that's what IC uses)
            inv_delta_phi = scaling_and_squaring(-delta_v, n_steps=self.n_steps)

            # INVERSE-COMPOSITION (left composition by inv_delta_phi):
            # phi_new = inv_delta_phi + warp(phi, inv_delta_phi)
            warped_phi = warp_displacement(phi, inv_delta_phi)  # sample phi at positions after inv_delta_phi
            phi = inv_delta_phi + warped_phi

            # update the warped mask by applying updated phi to the original mask (deferred warping)
            warped_mask = warp_vol_using_disp(orig_mask, phi)

            phis.append(phi)

        if return_params:
            return warped_mask, phis, phi
        return warped_mask

# ---------------------------
# Quick test
# ---------------------------
if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg_model = OmegaConf.create({
        "_target_": "monai.networks.nets.DynUNet",
        "in_channels": 2,   # will be concatenated vol + mask per iteration
        "out_channels": 3,  # predict 3D velocity (SVF)
        "spatial_dims": 3,
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "kernel_size": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        "upsample_kernel_size": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "filters": [32, 64, 128, 256, 320],
        "res_block": True,
        "norm_name": "INSTANCE",
        "deep_supervision": True,
        "deep_supr_num": 2
    })
    cfg = OmegaConf.create({
        "models": cfg_model,
        "max_v": 3.0,
        "n_steps": 5,
        "num_iters": 3,
        "max_delta_v": 0.6,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeformDynUnet(cfg).to(device)
    #model.eval()
    model.train()

    # fake data (use soft mask values in [0,1])
    x_img = torch.randn(1, 1, 64, 128, 128, device=device)
    soft_mask = torch.rand(1, 1, 64, 128, 128, device=device)  # soft mask in [0,1]
    x = torch.cat([x_img, soft_mask], dim=1)

    with torch.no_grad():
        warped_mask, phis, final_phi = model(x, return_params=True)

    print("warped_mask:", warped_mask.shape)
    print("num phis:", len(phis), "final_phi:", final_phi.shape)
