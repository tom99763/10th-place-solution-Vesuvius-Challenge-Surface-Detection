import logging
from typing import Callable, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from monai.metrics import DiceMetric
from torch import Tensor
from hydra.utils import instantiate
logger = logging.getLogger(__name__)

# default binary loss: BCE with logits
default_loss = nn.BCEWithLogitsLoss()


class ScrollSegmentorTrainer25D(L.LightningModule):
    """
    2.5D segmentation trainer.
    Follows the exact structure and conventions of ScrollSegmentorTrainer (3D),
    but adapted for 2D / 2.5D inputs of shape (B, C, H, W).
    """

    def __init__(
        self,
        model: torch.nn.Module = None,
        optimizer_factory: Callable =None,
        loss: Optional[Callable] = None,
        prediction_threshold: float = 0.5,
        scheduler_configs: Optional[dict] = None,
        dataset_name: Optional[str] = None,
        input_size: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        threshold: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "optimizer_factory", "loss"])

        self.model = model
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs or {}
        self.dataset_name = dataset_name

        # allow either prediction_threshold or legacy threshold argument
        self.prediction_threshold = prediction_threshold if threshold is None else threshold

        # Loss: default BCE if none provided
        self.loss = loss if loss is not None else default_loss

        # (2.5D does not use sliding-window inference)
        self.sliding_window_inferer = None

        # Dice metric for 2D binary segmentation
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean",
            get_not_nans=False,
        )

        logger.info(
            f"Initialized ScrollSegmentorTrainer25D -- dataset: {self.dataset_name} -- threshold: {self.prediction_threshold}"
        )

    # -----------------------------------
    # Helpers
    # -----------------------------------
    def _ensure_channel_first_mask(self, mask: Tensor) -> Tensor:
        """
        Ensure mask has channel dim = 1 as (B,1,H,W).
        Accepts (B,H,W) or (B,1,H,W).
        """
        if mask.ndim == 3:
            # (B,H,W) -> (B,1,H,W)
            return mask.unsqueeze(1)
        elif mask.ndim == 4 and mask.shape[1] == 3:
            return mask
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(mask.shape)}. "
                f"Expected (B,H,W) or (B,1,H,W)."
            )

    def _forward_logits(self, image: Tensor) -> Tensor:
        """
        2.5D forward pass.
        Always direct model forward (no sliding window).
        """
        return self.model(image)

    # -----------------------------------
    # Training / Validation / Test loops
    # -----------------------------------
    def training_step(self, batch, batch_idx):
        image, mask = batch

        # ensure mask has channel dim
        mask = self._ensure_channel_first_mask(mask).float()

        # forward
        pred_logits = self.model(image)  # expect (B,1,H,W)
        loss = self.loss(pred_logits, mask)

        # logging
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = self._ensure_channel_first_mask(mask).float()

        with torch.no_grad():
            pred_logits = self._forward_logits(image)
            loss = self.loss(pred_logits, mask)

            # convert logits -> prob -> binary
            pred_prob = torch.sigmoid(pred_logits)
            pred_bin = (pred_prob > self.prediction_threshold).float()

            # update dice metric
            self.dice_metric(y_pred=pred_bin, y=mask)

            # log
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return {"val_loss": loss.detach()}

    def on_validation_epoch_end(self):
        try:
            dice_score = self.dice_metric.aggregate().item()
        except Exception:
            dice_score = float("nan")
        self.log("val_dice", dice_score, prog_bar=True, sync_dist=True)
        self.dice_metric.reset()

    def test_step(self, batch, batch_idx):
        image, mask = batch
        mask = self._ensure_channel_first_mask(mask).float()

        with torch.no_grad():
            pred_logits = self._forward_logits(image)
            loss = self.loss(pred_logits, mask)

            pred_prob = torch.sigmoid(pred_logits)
            pred_bin = (pred_prob > self.prediction_threshold).float()

            self.dice_metric(y_pred=pred_bin, y=mask)
            self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        return {"test_loss": loss.detach()}

    def on_test_epoch_end(self):
        try:
            dice_score = self.dice_metric.aggregate().item()
        except Exception:
            dice_score = float("nan")
        self.log("test_dice", dice_score, prog_bar=True, sync_dist=True)
        self.dice_metric.reset()

    # -----------------------------------
    # Optimizer / Scheduler
    # -----------------------------------
    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_configs:
            schedulers = []
            for _, cfg in self.scheduler_configs.items():
                if cfg is None:
                    continue
                cfg["scheduler"] = cfg["scheduler"](optimizer=optimizer)
                schedulers.append(dict(cfg))
            return [optimizer], schedulers
        return optimizer