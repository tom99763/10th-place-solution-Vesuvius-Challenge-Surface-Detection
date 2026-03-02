import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from hydra.utils import instantiate


def load_sparse_z(path: Path, device=None):
    """Load sparse .npz and return dense tensor on given device."""
    d = np.load(path)
    idx = torch.tensor(d["idx"], dtype=torch.long)
    vals = torch.tensor(d["vals"], dtype=torch.float32)
    shape = tuple(d["shape"])

    D = torch.zeros(shape, dtype=vals.dtype, requires_grad=False)
    D[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = vals
    if device:
        D = D.to(device)
    return D


def reconstruct_mask(Z: torch.Tensor, D: torch.Tensor, upsample_factor=2):
    """
    Z: (B, K, D, H, W)
    D: (K, 1, k, k, k)
    Returns: mask (B, D*upsample, H*upsample, W*upsample)
    """
    sdf = to_sdf(Z, D)

    # Convert to binary mask
    mask = (sdf >= 0).float()  # (B, 1, D, H, W)

    # Upsample if needed
    if upsample_factor != 1:
        mask = F.interpolate(mask, scale_factor=upsample_factor, mode="nearest")

    return mask  # (B, 1, D', H', W')


def to_sdf(Z: torch.Tensor, D: torch.Tensor):
    """
    Z: (B, K, D, H, W)
    D: (K, 1, k, k, k)
    Returns: mask (B, D*upsample, H*upsample, W*upsample)
    """
    pad = D.shape[2] // 2
    sdf = F.conv3d(Z, D, padding=pad, groups=D.shape[0]).sum(1, keepdim=True) #(B, 1, D, H, W)
    return sdf

def from_sdf(sdf, upsample_factor = 2):
    # Convert to binary mask
    mask = (sdf >= 0).float()  # (B, 1, D, H, W)

    # Upsample if needed
    if upsample_factor != 1:
        mask = F.interpolate(mask, scale_factor=upsample_factor, mode="nearest")
    return mask


class ComposeCDL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictor = instantiate(cfg.models)
        self.cfg = cfg

        device = torch.device(cfg.device if hasattr(cfg, 'device') else 'cuda')
        self.D = load_sparse_z(Path(self.cfg.dictionary_path) / 'dictionary.npz', device=device)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        Returns:
            - training: reconstructed mask (B, 1, D, H, W)
            - eval: predicted Z_hat (B, K, D, H, W)
        """
        Z_hat = self.predictor(x)
        SDF_hat = to_sdf(Z_hat, self.D.detach())

        if self.training:
            return SDF_hat, Z_hat
        return SDF_hat