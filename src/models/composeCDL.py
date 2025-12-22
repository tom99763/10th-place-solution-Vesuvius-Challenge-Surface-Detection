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

    D = torch.zeros(shape, dtype=vals.dtype)
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
    B = Z.shape[0]
    pad = D.shape[2] // 2

    # Convolution with groups=K
    sdf = F.conv3d(Z, D, padding=pad, groups=D.shape[0]).sum(1)  # (B, D, H, W)

    # Convert to binary mask
    mask = (sdf >= 0).float()  # (B, D, H, W)

    # Upsample if needed
    if upsample_factor != 1:
        mask = F.interpolate(mask[:, None], scale_factor=upsample_factor, mode="nearest")

    return mask  # (B, 1, D', H', W')



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

        if self.training:
            prediction = reconstruct_mask(Z_hat, self.D)
            return prediction
        else:
            return Z_hat