import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate


class UNet25DModule(pl.LightningModule):
    def __init__(self, model_cfg, optimizer_cfg, dataloader_cfg):
        super().__init__()

        # Hydra configs
        self.model = instantiate(model_cfg)              # actual nn.Module
        self.optimizer_cfg = optimizer_cfg
        self.dataloader_cfg = dataloader_cfg

        # Assume you configure loss externally if needed
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return instantiate(self.optimizer_cfg, params=self.parameters())
