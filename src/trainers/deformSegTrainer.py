from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
import pytorch_lightning as pl
from src.trainers.losses import *
from monai.losses import DiceCELoss

# ---------------------
# Lightning Module
# ---------------------
class DiffeoRefineModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.automatic_optimization = True
        self.seg_loss = DiceCELoss(sigmoid=False, to_onehot_y=True, softmax=False,
                                   lambda_ce=cfg.lambda_ce, lambda_dice=cfg.lambda_dice)
        self.topo_loss = FastClDiceLoss(alpha=0.6)
        self.surf_loss = SurfaceLoss(tau_vox=2.0)

        # Validation accumulators (scalar averages)
        self.val_topo_losses = []
        self.val_surf_losses = []
        self.val_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        # For proper epoch-level averaging
        self.val_num_samples = 0

    def validation_step(self, batch, batch_idx):
        vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']
        x = torch.cat([vol, mask_oof], dim=1)

        # Replace this with real inference when ready
        # pred_warped = self.sliding_window_inferer(x, self.model)
        pred_warped = mask_oof  # ← your current baseline (OOF mask)

        ignore_mask = (mask != 2).float()
        target_mask = (mask > 0).float() * ignore_mask  # binary foreground

        # Apply ignore mask
        pred_masked = pred_warped * ignore_mask

        # === Compute losses exactly like training (but in eval mode) ===
        with torch.no_grad():
            topo_loss = self.topo_loss(pred_warped, target_mask)  # scalar
            surf_loss = self.surf_loss(pred_warped, target_mask)  # scalar

        # Binarize for DiceMetric
        pred_bin = pred_masked > self.cfg.threshold

        # Update MONAI Dice
        self.val_dice_metric(y_pred=pred_bin, y=(mask * ignore_mask).long())

        # Store losses weighted by batch size
        batch_size = vol.shape[0]
        self.val_topo_losses.append(topo_loss * batch_size)
        self.val_surf_losses.append(surf_loss * batch_size)
        self.val_num_samples += batch_size

        return None

    def on_validation_epoch_end(self):
        # === Average all losses over all samples in the epoch ===
        avg_topo_loss = torch.stack(self.val_topo_losses).sum() / self.val_num_samples
        avg_surf_loss = torch.stack(self.val_surf_losses).sum() / self.val_num_samples

        # Convert to metric scores (higher = better)
        topo_score = 1.0 - avg_topo_loss
        surf_score = 1.0 - avg_surf_loss

        # Final Dice from MONAI
        dice_score = self.val_dice_metric.aggregate().mean().item()

        # === Competition metric ===
        comp_metric = 0.30 * topo_score + 0.35 * surf_score + 0.35 * dice_score

        # === Logging ===
        self.log("val_dice", dice_score, prog_bar=True, rank_zero_only=True)
        self.log("val_topo_score", topo_score, prog_bar=True, rank_zero_only=True)
        self.log("val_surf_score", surf_score, prog_bar=True, rank_zero_only=True)
        self.log("val_comp_metric", comp_metric, prog_bar=True, rank_zero_only=True, sync_dist=True)

        if self.trainer.is_global_zero:
            print(f"\nVAL Epoch {self.current_epoch:03d} │ "
                  f"Dice: {dice_score:.4f} │ "
                  f"Topo: {topo_score:.4f} │ "
                  f"Surf: {surf_score:.4f} │ "
                  f"→ COMP: {comp_metric:.4f} ←\n")

        # === Reset everything ===
        self.val_dice_metric.reset()
        self.val_topo_losses.clear()
        self.val_surf_losses.clear()
        self.val_num_samples = 0

    # ----------------------------------------
    # Optimizer
    # ----------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
