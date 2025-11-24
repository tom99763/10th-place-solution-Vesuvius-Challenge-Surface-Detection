import logging
import os
from typing import Callable, Optional, Tuple
import monai
import torch
import torch.nn as nn
from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
from torch import Tensor
import pytorch_lightning as pl
from losses import *
import torch.nn.functional as F
from monai.losses import DiceCELoss

# ---------------------
# Lightning Module
# ---------------------
class ScrollDiffeoRefineModule(pl.LightningModule):
    def __init__(self, model, cfg ):
        super().__init__()
        self.model = model
        self.lr = cfg.lr
        self.lambda_jac = cfg.lambda_jac
        self.lambda_smooth = cfg.lambda_smooth

        self.seg_loss = DiceCELoss(
            sigmoid=True,
            to_onehot_y=False,
            softmax=False,
            reduction="mean",
            squared_pred=True,
            lambda_ce=0.1,
            lambda_dice=0.9,
        )

    def forward(self, x):
        return self.model(x, return_params=True)

    # ----------------------------------------
    # Smoothness: ||∇v||² on the SVF
    # ----------------------------------------
    def svf_smoothness(self, v):
        return (
            (v[:,:,1:] - v[:,:,:-1]).pow(2).mean() +
            (v[:,:,:,1:] - v[:,:,:,:-1]).pow(2).mean() +
            (v[:,:,:,:,1:] - v[:,:,:,:,:-1]).pow(2).mean()
        ) / 3.0

    # ----------------------------------------
    # TRAINING STEP
    # ----------------------------------------
    def training_step(self, batch, batch_idx):
        x = torch.cat([batch["Image"], batch["Mask_OOF"]], dim=1)

        pred_warped, v, phi = self(x, return_params=True)
        gt = batch["Mask"]

        # segmentation loss using MONAI DiceCE
        L_seg = self.seg_loss(pred_warped, gt)

        # smoothness regularizer
        L_smooth = self.svf_smoothness(v)

        # jacobian folding penalty
        det = jacobian_determinant(phi)
        L_jac = torch.relu(-det).mean()

        loss = L_seg + self.lambda_smooth * L_smooth + self.lambda_jac * L_jac

        self.log("loss", loss, prog_bar=True)
        self.log("seg", L_seg)
        self.log("jac", L_jac)
        self.log("smooth", L_smooth)

        return loss

    # ----------------------------------------
    # VALIDATION STEP
    # ----------------------------------------
    def validation_step(self, batch, batch_idx):
        pass

    # ----------------------------------------
    # Optimizer
    # ----------------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
