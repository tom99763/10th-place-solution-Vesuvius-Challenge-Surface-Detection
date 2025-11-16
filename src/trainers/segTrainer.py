import logging
import os
from typing import Callable, Optional, Tuple

import lightning as L
import monai
import torch
import torch.nn as nn
from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
from torch import Tensor

logger = logging.getLogger(__name__)

# default binary loss: BCE with logits (more stable than sigmoid + BCE)
default_loss = nn.BCEWithLogitsLoss()


class ScrollSegmentorTrainer(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_factory: Callable,
        loss: Optional[Callable] = None,
        prediction_threshold: float = 0.5,
        scheduler_configs: Optional[dict] = None,
        dataset_name: Optional[str] = None,
        input_size: Optional[Tuple[int, int, int]] = None,
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

        # Loss: default to BCEWithLogitsLoss if not provided
        self.loss = loss if loss is not None else default_loss

        # Sliding window inferer (for validation / test large volumes)
        if input_size is None or batch_size is None:
            # If not provided, do not use sliding window inferer — use direct model forward
            self.sliding_window_inferer = None
        else:
            self.sliding_window_inferer = SlidingWindowInfererAdapt(
                roi_size=input_size, sw_batch_size=batch_size, overlap=0.5
            )

        # Dice metric for binary segmentation.
        # For binary we use include_background=True and expect (B,1,D,H,W) binary tensors.
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean",
            get_not_nans=False,  # avoid NaN issues when empty masks appear
        )

        logger.info(f"Initialized RSNABinaryModule -- dataset: {self.dataset_name} -- threshold: {self.prediction_threshold}")

    # -------------------------
    # Helpers
    # -------------------------
    def _ensure_channel_first_mask(self, mask: Tensor) -> Tensor:
        """
        Ensure mask has channel dim = 1 as shape (B,1,D,H,W).
        Accepts masks shaped (B,D,H,W) or (B,1,D,H,W).
        """
        if mask.ndim == 4:
            # (B,D,H,W) -> (B,1,D,H,W)
            return mask.unsqueeze(1)
        elif mask.ndim == 5 and mask.shape[1] == 1:
            return mask
        else:
            raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}. Expected (B,D,H,W) or (B,1,D,H,W).")

    def _forward_logits(self, image: Tensor) -> Tensor:
        """
        Run model forward. If a sliding-window inferer was provided, use it.
        Otherwise call the model directly.
        Returns logits (B,1,D,H,W)
        """
        if self.sliding_window_inferer is not None:
            return self.sliding_window_inferer(image, self.model)
        else:
            return self.model(image)

    # -------------------------
    # Training / Validation / Test
    # -------------------------
    def training_step(self, batch, batch_idx):
        image, mask = batch
        # ensure mask has channel dim
        mask = self._ensure_channel_first_mask(mask).float()

        # forward
        pred_logits = self.model(image)  # expect (B,1,D,H,W)
        loss = self.loss(pred_logits, mask)

        # logging
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = self._ensure_channel_first_mask(mask).float()

        with torch.no_grad():
            pred_logits = self._forward_logits(image)  # (B,1,D,H,W)
            loss = self.loss(pred_logits, mask)

            # convert logits -> probability -> binary prediction for metric
            pred_prob = torch.sigmoid(pred_logits)
            pred_bin = (pred_prob > self.prediction_threshold).float()

            # update dice metric (expects tensors on the same device)
            self.dice_metric(y_pred=pred_bin, y=mask)

            # logging
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"val_loss": loss.detach()}

    def on_validation_epoch_end(self):
        # aggregate and log dice
        try:
            dice_score = self.dice_metric.aggregate().item()
        except Exception:
            # if aggregate fails (e.g., no examples), set to 0.
            dice_score = float("nan")
        self.log("val_dice", dice_score, prog_bar=True, sync_dist=True)
        self.dice_metric.reset()

    def test_step(self, batch, batch_idx):
        # similar to validation_step but logs test metrics
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

    # -------------------------
    # Optimizer / Scheduler
    # -------------------------
    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_configs:
            schedulers = []
            for _, cfg in self.scheduler_configs.items():
                if cfg is None:
                    continue
                # Expect cfg to contain a "scheduler" callable that returns a scheduler when called with optimizer
                # Similar to your previous pattern.
                cfg["scheduler"] = cfg["scheduler"](optimizer=optimizer)
                schedulers.append(dict(cfg))
            return {"optimizer": optimizer, "lr_scheduler": schedulers}
        return optimizer