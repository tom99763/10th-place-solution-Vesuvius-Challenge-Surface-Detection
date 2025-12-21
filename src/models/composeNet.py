# Retry with corrected UNet upsampling (uses trilinear interpolation to avoid size mismatches).
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
import numpy as np

class MaskDecompositionTorch:
    """
    Exact scroll mask decomposition in PyTorch.
    Input: mask (B, 1, D, H, W) binary
    Returns: dict with sdf, normals, thickness (all torch tensors)
    """
    def __init__(self, substeps=5):
        self.substeps = substeps

    def inverse_transform(self, pred):
        """
        pred: dict with 'sdf', 'normals', 'thickness'
        Returns: reconstructed mask (B,1,D,H,W)
        """
        sdf = pred["sdf"]
        normals = pred["normals"]
        thickness = pred["thickness"]
        B, _, D, H, W = sdf.shape
        device = sdf.device

        recon = torch.zeros((B, 1, D, H, W), dtype=torch.float32, device=device)

        for b in range(B):
            center_mask = sdf[b,0] <= 0
            coords = center_mask.nonzero(as_tuple=False)  # (N,3) z,y,x

            if coords.numel() == 0:
                continue

            dz = normals[b,0][center_mask]
            dy = normals[b,1][center_mask]
            dx = normals[b,2][center_mask]
            t_half = thickness[b,0][center_mask] / 2.0

            steps = torch.linspace(-1, 1, self.substeps, device=device)[None, :] * t_half[:, None]

            z_coords = coords[:, 0:1].float() + dz[:, None] * steps
            y_coords = coords[:, 1:2].float() + dy[:, None] * steps
            x_coords = coords[:, 2:3].float() + dx[:, None] * steps

            z_idx = torch.clamp(z_coords.round().long(), 0, D-1).ravel()
            y_idx = torch.clamp(y_coords.round().long(), 0, H-1).ravel()
            x_idx = torch.clamp(x_coords.round().long(), 0, W-1).ravel()

            recon[b, 0, z_idx, y_idx, x_idx] = 1

        return recon


class ResBlock3D(nn.Module):
    def __init__(self, channels, norm=nn.InstanceNorm3d):
        super().__init__()

        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = norm(channels, affine=True)

        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = norm(channels, affine=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + identity)


class ResHead3D(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=3, norm=nn.InstanceNorm3d):
        super().__init__()

        self.blocks = nn.Sequential(
            *[ResBlock3D(in_channels, norm=norm) for _ in range(n_blocks)]
        )

        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.blocks(x)
        return self.out_conv(x)


class ComposeNet3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.predictor = instantiate(cfg.models)

        C = cfg.models.out_channels
        norm = nn.InstanceNorm3d

        self.thickness_head = ResHead3D(
            in_channels=C,
            out_channels=1,
            n_blocks=cfg.n_blocks,
            norm=norm,
        )

        self.sdf_head = ResHead3D(
            in_channels=C,
            out_channels=1,
            n_blocks=cfg.n_blocks,
            norm=norm,
        )

        self.normals_head = ResHead3D(
            in_channels=C,
            out_channels=3,
            n_blocks=cfg.n_blocks,
            norm=norm,
        )

        self.decomp = MaskDecompositionTorch()

    def forward(self, x, return_components = False):
        fmap = self.predictor(x)
        thicknesses = self.thickness_head(fmap)
        sdfs = self.sdf_head(fmap)
        normals = self.normals_head(fmap)
        components = {"sdf": sdfs, "normals": normals, "thickness": thicknesses}
        if return_components:
            return components
        return self.decompose(components)

    def decompose(self, components):
        return self.decomp.inverse_transform(pred=components)