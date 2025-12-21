import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
import numpy as np
from pathlib import Path

class ComposeCDL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictor = instantiate(cfg.models)
        self.cfg = cfg
    def forward(self, x):
        Z_hat = self.predictor(x)
        return Z_hat