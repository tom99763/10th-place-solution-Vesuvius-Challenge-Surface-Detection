# Retry with corrected UNet upsampling (uses trilinear interpolation to avoid size mismatches).
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

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

        C = cfg.head.in_channels
        norm = nn.InstanceNorm3d

        self.skeleton_head = ResHead3D(
            in_channels=C,
            out_channels=cfg.head.skeleton_out,
            n_blocks=cfg.head.n_blocks,
            norm=norm,
        )

        self.edge_head = ResHead3D(
            in_channels=C,
            out_channels=cfg.head.edge_out,
            n_blocks=cfg.head.n_blocks,
            norm=norm,
        )

        self.cover_head = ResHead3D(
            in_channels=C,
            out_channels=cfg.head.cover_out,
            n_blocks=cfg.head.n_blocks,
            norm=norm,
        )

    def forward(self, x, return_components = False):
        fmap = self.predictor(x)
        skeleton = self.skeleton_head(fmap)
        edge = self.edge_head(fmap)
        cover = self.cover_head(fmap)
        prediction = skeleton.sigmoid() + edge.sigmoid() + cover.sigmoid()
        if return_components:
            components = {
                "skeleton": skeleton,
                "edge": edge,
                "cover": cover
            }
            return prediction, components
        return prediction