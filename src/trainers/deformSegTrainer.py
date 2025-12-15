from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
import pytorch_lightning as pl
from src.trainers.losses import *
from monai.losses import DiceCELoss
from src.procs.proc_data import *

# ---------------------
# Lightning Module
# ---------------------
class DiffeoRefineModule(pl.LightningModule):
    def __init__(self, model, cfg ):
        super().__init__()
        self.model = model
        self.lr = cfg.lr
        self.lambda_jac = cfg.lambda_jac
        self.lambda_smooth = cfg.lambda_smooth
        self.cfg = cfg

        # Loss
        self.seg_loss = DiceCELoss(
            sigmoid=False,  # ← critical fix
            to_onehot_y=True,  # because your labels are integer class indices
            softmax=False,
            reduction="mean",
            squared_pred=True,
            lambda_ce=cfg.lambda_ce,
            lambda_dice=cfg.lambda_dice,
        )
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=cfg.input_size, sw_batch_size=2, overlap=0, mode="constant"
        )
        self.topo_loss = FastClDiceLoss()  # ← main topology driver
        self.soft_surf_loss = SoftSDFLoss()
        self.surf_loss = SurfaceLoss(tau_vox=2.0)  # ← main SurfaceDice driver

        # Validation accumulators (scalar averages)
        self.val_topo_losses = []
        self.val_surf_losses = []
        self.val_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        # For proper epoch-level averaging
        self.val_num_samples = 0

    def forward(self, x, return_params=False):
        return self.model(x, return_params=return_params)

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
        vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']
        if self.cfg.apply_gaussian:
            x = torch.cat([vol, gaussian_blur_3d(mask_oof)], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)
        pred_warped, v, phi = self(x, return_params=True)
        ignore_mask = mask != 2

        # segmentation loss using MONAI DiceCE
        L_seg = self.seg_loss(pred_warped * ignore_mask, mask * ignore_mask)
        L_topo = self.topo_loss(pred_warped * ignore_mask, mask * ignore_mask)
        L_surf = self.soft_surf_loss(pred_warped * ignore_mask, mask * ignore_mask)

        # smoothness regularizer
        L_smooth = self.svf_smoothness(v)

        # jacobian folding penalty
        # det = jacobian_determinant(phi)
        # L_jac = torch.relu(-det).mean()
        L_jac = jacobian_log_barrier(phi)

        loss = L_seg + 0.5 * L_topo + 0.5 * L_surf + self.lambda_smooth * L_smooth + self.lambda_jac * L_jac

        self.log("loss", loss, prog_bar=True)
        self.log("seg", L_seg, prog_bar=True)
        self.log("jac", L_jac, prog_bar=True)
        self.log("smooth", L_smooth, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']
        if self.cfg.apply_gaussian:
            x = torch.cat([vol, gaussian_blur_3d(mask_oof)], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)

        # Replace this with real inference when ready
        pred_warped = self.sliding_window_inferer(x, self.model)
        #pred_warped = mask_oof  # ← your current baseline (OOF mask)

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
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=1e-2)


class ICDiffeoRefineModule(DiffeoRefineModule):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)