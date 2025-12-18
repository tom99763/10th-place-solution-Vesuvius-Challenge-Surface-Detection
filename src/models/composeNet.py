# Retry with corrected UNet upsampling (uses trilinear interpolation to avoid size mismatches).
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

class ComposeNet3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictor = instantiate(cfg.models)

    def forward(self, x):
        prediction = self.predictor(x)
        if not self.training:
            mask = x[:, 1:2]
            corr = torch.sign(prediction)
            mask = (mask + corr).clamp(0, 1)
            return mask
        return prediction