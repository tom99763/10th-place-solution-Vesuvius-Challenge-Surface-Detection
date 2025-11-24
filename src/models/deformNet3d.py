# Retry with corrected UNet upsampling (uses trilinear interpolation to avoid size mismatches).
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
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
        #det = jacobian_determinant(phi)

    def forward(self, x, return_params = True):
        #x: (batch, 2, d, h, w)
        raw_v = self.predictor(x)
        v = torch.tanh(raw_v) * self.cfg.max_v
        phi = scaling_and_squaring(v, n_steps=self.cfg.n_steps)
        soft_oof = x[:, 1:2, :, :, :]
        warped = warp_vol_using_disp(soft_oof, phi)
        if return_params:
            return warped, v, phi
        return warped

# Lightweight forward test
if __name__ == "__main__":
    pass
