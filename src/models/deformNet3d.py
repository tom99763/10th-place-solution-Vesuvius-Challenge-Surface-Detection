# Retry with corrected UNet upsampling (uses trilinear interpolation to avoid size mismatches).
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

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


def create_residual_unet(
        in_channels=2,
        out_channels=3,
        channels=(32, 64, 128, 256, 320, 320),
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        deep_supervision=False,
):
    # Number of stages in decoder is len(channels) - 1
    n_conv_per_stage_decoder = [1] * (len(channels) - 1)

    model = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=len(channels),
        features_per_stage=channels,
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=out_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    return model


class DeformDynUnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.use_resenc:
            self.predictor = create_residual_unet()
        else:
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
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_iters = getattr(cfg, "num_iters", 2)
        self.max_delta_v = getattr(cfg, "max_delta_v", 0.6)
        self.n_steps = getattr(cfg, "n_steps", 4)  # ↓ from 6

    def forward(self, x, return_params=False):
        """
        Optimized IC forward:
        - Incremental mask warping
        - Fewer exp calls
        - No redundant recomputation
        """
        vol, mask = x[:, 0:1], x[:, 1:2]
        B, _, D, H, W = mask.shape
        device = mask.device

        # identity deformation
        phi = torch.zeros(B, 3, D, H, W, device=device, dtype=mask.dtype)

        warped_mask = mask  # incremental warp
        phis = []

        for _ in range(self.num_iters):
            # predictor input
            inp = torch.cat([vol, warped_mask], dim=1)

            # predict incremental SVF
            delta_v = torch.tanh(self.predictor(inp)) * self.max_delta_v

            # exp(-Δv)
            inv_delta_phi = scaling_and_squaring(
                -delta_v, n_steps=self.n_steps
            )

            # φ ← Δφ⁻¹ ∘ φ
            phi = inv_delta_phi + warp_displacement(phi, inv_delta_phi)

            # incremental mask warp (FAST)
            warped_mask = warp_vol_using_disp(warped_mask, inv_delta_phi)

            if return_params:
                phis.append(phi)

        if return_params:
            return warped_mask, phis, phi
        return warped_mask

# ---------------------------
# defromnetv2
# ---------------------------

def soft_sdf(x, eps=1e-4):
    # x in [0,1]
    return torch.log(x + eps) - torch.log(1 - x + eps)

class TopoFix(nn.Module):
    def __init__(self, max_offset=2.0):
        super().__init__()
        self.max_offset = max_offset

    def forward(self, warped_mask, topo_gate):
        """
        warped_mask: (B,1,D,H,W) in [0,1]
        topo_gate:   (B,1,D,H,W) in [0,1]
        """
        sdf = soft_sdf(warped_mask)
        delta = self.max_offset * torch.tanh(topo_gate)
        sdf_corr = sdf + delta * topo_gate
        corrected = torch.sigmoid(sdf_corr)
        return corrected


class DeformDynUnetV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictor = create_residual_unet(
            in_channels=2,
            out_channels=4
        )
        self.max_v = cfg.max_v
        self.topofix = TopoFix(max_offset=cfg.max_topo_offset)

    def forward(self, x, return_params=False):
        raw = self.predictor(x)
        # split
        raw_v = raw[:, :3]
        raw_t = raw[:, 3:4]
        # SVF
        v = torch.tanh(raw_v) * self.max_v
        phi = scaling_and_squaring(v)
        # warp
        warped = warp_vol_using_disp(x[:,1:2], phi)
        # topology gate
        t = torch.sigmoid(raw_t)
        # topo fix
        corrected = self.topofix(warped, t)
        if return_params:
            return corrected, v, phi, t
        return corrected
