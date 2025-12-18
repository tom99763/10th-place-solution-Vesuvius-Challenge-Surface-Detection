# Retry with corrected UNet upsampling (uses trilinear interpolation to avoid size mismatches).
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

class ComposeNet3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictor = instantiate(cfg.models)
        self.eps = cfg.eps

    def forward(self, x):
        prediction = self.predictor(x)
        prediction = torch.tanh(prediction)
        if not self.training:
            mask = x[:, 1:2]
            corr = torch.where(
                prediction > self.eps, 1,
                torch.where(prediction < -self.eps, -1, 0)
            )
            mask = (mask + corr).clamp(0, 1)
            return mask
        return prediction