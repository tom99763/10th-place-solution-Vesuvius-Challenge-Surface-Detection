import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
import numpy as np
from pathlib import Path


def load_sparse_tensor(path: str) -> np.ndarray:
    """
    Load a sparse tensor saved in index+value+shape format
    Returns dense np.ndarray
    """
    data = np.load(path)
    tensor = np.zeros(data['shape'], dtype=np.float32)
    tensor[data['idx']] = data['values']
    return tensor

def reconstruct_mask(Z, D):
    K = D.shape[0]
    pad = D.shape[2] // 2
    sdf_hat = F.conv3d(Z, D, padding=pad, groups=K).sum(dim=1, keepdim=True)
    mask_hat = sdf_hat>=0
    return mask_hat

class ComposeCDL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictor = instantiate(cfg.models)
        self.cfg = cfg
        self.D = torch.from_numpy(load_sparse_tensor(str(Path(self.cfg.dictionary_path)/'dictionary.npz')))
    def forward(self, x):
        Z_hat = self.predictor(x)
        if not self.training:
            D = self.D.to(Z_hat.device)
            return reconstruct_mask(Z_hat, D)
        return Z_hat