import torch
from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
import pytorch_lightning as pl
from src.trainers.losses import *
from monai.losses import DiceCELoss, TverskyLoss
from src.procs.proc_data import *
import sys
sys.path.append('../../')
from cv import calc_score
from monai.metrics import DiceMetric
from utils import EMA


class RefineNetModule(pl.LightningModule):
    def __init__(self, model, cfg ):
        super().__init__()
        self.model = model
        self.lr = cfg.lr
        self.lambda_jac = cfg.lambda_jac
        self.lambda_smooth = cfg.lambda_smooth
        self.cfg = cfg

        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=cfg.input_size, sw_batch_size=2, overlap=cfg.overlap, mode="gaussian"
        )

        # Loss functions
        self.ce_loss = (TopKCrossEntropyLoss(top_k_percent=cfg.ce_top_k, ignore_index=cfg.ignore_index)
                        if cfg.ce_top_k < 1.0 else nn.CrossEntropyLoss(ignore_index=cfg.ignore_index))
        self.dice_loss = DiceLoss(smooth=1e-5, ignore_index=cfg.ignore_index)
        self.surface_dice_loss = SurfaceDiceLoss(
            ignore_label=cfg.ignore_index, soft_skel_iterations=cfg.surface_dice_iterations, smooth=1.0
        )
        self.skeleton_loss = SkeletonRecallLoss(ignore_index=cfg.ignore_index)

        # Validation accumulators (scalar averages)
        self.scores = []
        self.topo_scores = []
        self.voi_scores = []
        self.surface_scores = []

        # EMA (optional)
        self.ema = None
        if cfg.use_ema:
            self.ema = EMA(self.model, decay=cfg.ema_decay, warmup=cfg.ema_warmup)

        # Validation metric
        self.val_dice_scores = []

        # For proper epoch-level averaging
        self.val_num_samples = 0

    def forward(self, x):
        # Use EMA for inference, original model for training
        model = self.ema.module if (self.ema is not None and not self.training) else self.model
        out = model(x)
        if not self.training and isinstance(out, (list, tuple)):
            return out[0]
        return out

    # ----------------------------------------
    # TRAINING STEP
    # ----------------------------------------
    def _get_loss_weights(self):
        """Get loss weights with backward compatibility for multiple formats."""
        weights = self.cfg.loss_weights

        # Backwards compat:
        # - (ce, dice) => no surface losses
        # - (ce, dice, surface_dice) => SurfaceDice only
        # - (ce, dice, surface_dice, skeleton_recall) => All
        if len(weights) == 2:
            return weights[0], weights[1], 0.0, 0.0
        if len(weights) == 3:
            return weights[0], weights[1], weights[2], 0.0
        if len(weights) == 4:
            return weights[0], weights[1], weights[2], weights[3]
        return weights[0], weights[1], weights[2], 0.0

    def _downsample_labels(self, labels, target_size):
        """Downsample labels to target size using nearest neighbor interpolation.

        Args:
            labels: (B, D, H, W) - class indices or binary masks
            target_size: Tuple of (D, H, W)

        Returns:
            Downsampled labels of shape (B, *target_size)
        """
        if labels.shape[1:] == target_size:
            return labels

        # Ensure input is 5D for interpolation (B, 1, D, H, W)
        if labels.dim() == 4:
            labels_5d = labels.unsqueeze(1).float()
        else:
            # If already 5D, assume channel is at index 1
            labels_5d = labels.float()

        # Downsample using nearest neighbor (preserves class indices)
        labels_downsampled = F.interpolate(
            labels_5d,
            size=target_size,
            mode='nearest'
        )
        # Return as 4D (B, D, H, W) for CrossEntropy
        return labels_downsampled.squeeze(1)

    def _safe_loss(self, loss_val, logits):
        """Replace NaN losses with small regularization term."""
        return (logits ** 2).mean() * 1e-8 if torch.isnan(loss_val) else loss_val

    def _compute_single_scale_loss(self, logits, labels, skeletons):
        """Compute loss at a single scale."""
        # logits, labels, skeletons = self._prepare_tensors(
        #     logits, labels, skeletons)
        ce_w, dice_w, surf_w, skel_w = self._get_loss_weights()

        loss_ce = self._safe_loss(self.ce_loss(logits, labels.long()), logits)
        loss_dice = self._safe_loss(
            self.dice_loss(logits, labels.long()), logits)

        # Surface Dice (only if weight > 0)
        if surf_w > 0:
            logits_binary = logits[:, 1:2] - logits[:, 0:1]
            loss_surf_dice = self.surface_dice_loss(
                logits_binary, labels.unsqueeze(1)).mean()
            loss_surf_dice = self._safe_loss(loss_surf_dice, logits)
        else:
            loss_surf_dice = loss_ce * 0.0  # Zero loss with grad_fn

        # Skeleton Recall (only if weight > 0 and skeletons provided)
        if skel_w > 0 and skeletons is not None:
            loss_skel = self.skeleton_loss(logits, skeletons, labels)
        else:
            loss_skel = loss_ce * 0.0  # Zero loss with grad_fn

        loss = ce_w * loss_ce + dice_w * loss_dice + \
               surf_w * loss_surf_dice + skel_w * loss_skel
        return {'loss': loss, 'loss_ce': loss_ce, 'loss_dice': loss_dice,
                'loss_surf_dice': loss_surf_dice, 'loss_skel': loss_skel}

    def _log_losses(self, prefix, losses, on_step=False):
        """Log loss components and metrics."""
        self.log(f'{prefix}/loss', losses['loss'],
                 on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/loss_ce',
                 losses['loss_ce'], on_step=False, on_epoch=True)
        self.log(f'{prefix}/loss_dice',
                 losses['loss_dice'], on_step=False, on_epoch=True)
        self.log(f'{prefix}/loss_surf_dice',
                 losses['loss_surf_dice'], on_step=False, on_epoch=True)
        self.log(f'{prefix}/loss_skel',
                 losses['loss_skel'], on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        vol, mask, prob_mask_oof, skel = (
            batch["Image"],
            batch["Mask"],
            batch["Mask_OOF"],
            batch["Skel"],
        )
        mask_oof = (prob_mask_oof > 0.3).float()
        x = torch.cat([vol, mask_oof], dim=1)
        logits = self(x)
        losses = self._compute_single_scale_loss(logits, mask, skel)
        self._log_losses('train', losses, on_step=True)
        return None

    def validation_step(self, batch, batch_idx):
        vol, mask, prob_mask_oof, skel = (
            batch["Image"],
            batch["Mask"],
            batch["Mask_OOF"],
            batch["Skel"],
        )
        valid_mask = mask != self.cfg.ignore_label

        mask_oof = (prob_mask_oof > 0.3).float()
        x = torch.cat([vol, mask_oof], dim=1)
        prediction = self.sliding_window_inferer(x, self.model)
        prediction = prediction.argmax(dim=1, keepdims=True) * valid_mask
        mask *= valid_mask

        # print(prediction)
        score = calc_score(mask.cpu().numpy()[0, 0], prediction.cpu().numpy()[0, 0])
        self.val_num_samples += x.shape[0]
        self.scores.append(score.score)
        self.topo_scores.append(score.topo.toposcore)
        self.voi_scores.append(score.voi.voi_score)
        self.surface_scores.append(score.surface_dice)
        return None

    def on_before_zero_grad(self, optimizer):
        """Update EMA after optimizer step, before zeroing gradients."""
        if self.ema is not None:
            self.ema.update(self.model)

    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log(
            "lr",
            lr,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        comp_score = np.stack(self.scores).sum() / self.val_num_samples
        topo_score = np.stack(self.topo_scores).sum() / self.val_num_samples
        voi_score = np.stack(self.voi_scores).sum() / self.val_num_samples
        surface_score = np.stack(self.surface_scores).sum() / self.val_num_samples
        bias_comp_score = 0.5 * voi_score + 0.5 * surface_score

        # === Logging ===
        self.log("val_topo", topo_score, prog_bar=True, rank_zero_only=True)
        self.log("val_voi", voi_score, prog_bar=True, rank_zero_only=True)
        self.log("val_surface", surface_score, prog_bar=True, rank_zero_only=True)
        self.log("val_comp_metric", comp_score, prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log("val_bias_comp_metric", bias_comp_score, prog_bar=True, rank_zero_only=True, sync_dist=True)

        if self.trainer.is_global_zero:
            print(f"\nVAL Epoch {self.current_epoch:03d} │ "
                  f"VOI: {voi_score:.4f} │ "
                  f"Topo: {topo_score:.4f} │ "
                  f"Surf: {surface_score:.4f} │ "
                  f"→ COMP: {comp_score:.4f} ←\n"
                  f"→bias comp: {bias_comp_score: .4f}←\n"
                  )

        # === Reset everything ===
        self.scores = []
        self.topo_scores = []
        self.voi_scores = []
        self.surface_scores = []
        self.val_num_samples = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-2,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.75,
            patience=5,
            threshold=1e-4,
            cooldown=2,
            min_lr=1e-4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
