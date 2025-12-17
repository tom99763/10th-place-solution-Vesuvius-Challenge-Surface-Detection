from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
import pytorch_lightning as pl
from networkx.classes import edges

from src.trainers.losses import *
from monai.losses import DiceCELoss
from src.procs.proc_data import *

# ---------------------
# Lightning Module
# ---------------------
class ComposeRefineModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg.lr

        # === Base segmentation loss (binary heads) ===
        self.dice_ce = DiceCELoss(
            sigmoid=True,
            softmax=False,
            to_onehot_y=False,
            squared_pred=True,
            lambda_ce=cfg.lambda_ce,
            lambda_dice=cfg.lambda_dice,
        )

        self.dice_ce_n = DiceCELoss(
            sigmoid=False,
            softmax=False,
            to_onehot_y=False,
            squared_pred=True,
            lambda_ce=cfg.lambda_ce,
            lambda_dice=cfg.lambda_dice,
        )
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=cfg.input_size, sw_batch_size=2, overlap=0, mode="constant"
        )

        # === Topology & surface ===
        self.cldice = FastClDiceLoss()
        self.sdf = SoftSDFLoss(tau_vox=2.0)
        self.surface = SurfaceLoss(tau_vox=2.0)

        # Validation accumulators (scalar averages)
        self.val_topo_losses = []
        self.val_surf_losses = []
        self.val_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        # For proper epoch-level averaging
        self.val_num_samples = 0

    def forward(self, x, return_components=False):
        return self.model(x, return_components=return_components)

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        vol, mask_oof, mask = batch["Image"], batch["Mask_OOF"], batch["Mask"]
        skel_gt = batch["Skeleton"]
        edge_gt = batch["Edge"]
        cover_gt = batch["Cover"]
        ignore_mask = (mask != 2).float()

        if self.cfg.apply_gaussian:
            x = torch.cat([vol, gaussian_blur_3d(mask_oof)], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)

        _, comps = self(x, return_components=True)

        skel_pred = comps["Skeleton"]
        edge_pred = comps["Edge"]
        cover_pred = comps["Cover"]

        # === Skeleton loss ===
        L_skel = (
            self.dice_ce(skel_pred * ignore_mask, skel_gt * ignore_mask)
            + self.cfg.lambda_topo * self.cldice(
                skel_pred.sigmoid() * ignore_mask, skel_gt * ignore_mask
            )
        )

        # === Edge loss ===
        L_edge = (
            self.dice_ce(edge_pred * ignore_mask, edge_gt * ignore_mask)
            + self.cfg.lambda_sdf * self.sdf(
                edge_pred.sigmoid() * ignore_mask, edge_gt * ignore_mask
            )
        )

        # === Cover loss ===
        L_cover = self.dice_ce(cover_pred * ignore_mask, cover_gt * ignore_mask)

        # === Optional recomposition consistency ===
        if self.cfg.lambda_consistency > 0:
            recon = (
                skel_pred.sigmoid()
                + edge_pred.sigmoid()
                + cover_pred.sigmoid()
            ).clamp(0, 1)

            gt_full = (skel_gt + edge_gt + cover_gt).clamp(0, 1)
            L_cons = self.dice_ce_n(recon * ignore_mask, gt_full * ignore_mask)
        else:
            L_cons = 0.0

        loss = L_skel + L_edge + L_cover + self.cfg.lambda_consistency * L_cons

        self.log_dict({
            "loss": loss,
            "loss_skel": L_skel,
            "loss_edge": L_edge,
            "loss_cover": L_cover,
        }, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']
        if self.cfg.apply_gaussian:
            x = torch.cat([vol, gaussian_blur_3d(mask_oof)], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)

        prediction = self.sliding_window_inferer(x, self.model)

        ignore_mask = (mask != 2).float()
        target_mask = (mask > 0).float() * ignore_mask  # binary foreground

        # Apply ignore mask
        pred_masked = prediction * ignore_mask

        # === Compute losses exactly like training (but in eval mode) ===
        with torch.no_grad():
            topo_loss = self.cldice(prediction, target_mask)  # scalar
            surf_loss = self.surface(prediction, target_mask)  # scalar

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

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-2
        )