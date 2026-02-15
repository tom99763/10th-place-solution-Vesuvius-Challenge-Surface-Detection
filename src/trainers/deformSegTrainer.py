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
from src.models.deformNet3d import *

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
            roi_size=cfg.input_size, sw_batch_size=2, overlap=0.5, mode="gaussian"
        )
        self.topo_loss = FastClDiceLoss()  # ← main topology driver
        self.soft_surf_loss = SoftSDFLoss()
        self.surf_loss = SurfaceLoss(tau_vox=2.0)  # ← main SurfaceDice driver
        self.skel_loss = SkeletonRecallLoss()

        # Validation accumulators (scalar averages)
        self.scores = []
        self.topo_scores = []
        self.voi_scores = []
        self.surface_scores = []

        # Validation metric
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean",
            get_not_nans=False,
        )

        self.val_dice_scores = []

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
        vol, mask, mask_oof, skel = batch['Image'], batch['Mask'], batch['Mask_OOF'], batch['Skel']

        # random threhsold augmentation
        if self.cfg.is_prob_oof_mask:
            threshold = torch.empty(1, device=mask_oof.device).uniform_(0.1, 0.5)
            mask_oof = (mask_oof > threshold).float()

        if self.cfg.apply_gaussian:
            mask_oof = gaussian_blur_3d(mask_oof, self.cfg.kernel_size, self.cfg.sigma)

        x = torch.cat([vol, mask_oof], dim=1)
        pred_warped, v, phi = self(x, return_params=True)
        ignore_mask = mask != 2

        # segmentation loss using MONAI DiceCE
        L_seg = self.seg_loss(pred_warped * ignore_mask, mask * ignore_mask)
        L_skel = self.skel_loss(pred_warped, skel, mask)
        L_topo = self.topo_loss(pred_warped * ignore_mask, mask * ignore_mask)
        L_surf = self.soft_surf_loss(pred_warped * ignore_mask, mask * ignore_mask)

        # smoothness regularizer
        L_smooth = self.svf_smoothness(v)

        # jacobian folding penalty
        # det = jacobian_determinant(phi)
        # L_jac = torch.relu(-det).mean()
        L_jac = jacobian_log_barrier(phi)

        loss = L_seg + 0.1 * L_skel + 0.5 * L_topo + 0.5 * L_surf + self.lambda_smooth * L_smooth + self.lambda_jac * L_jac

        self.log("loss", loss, prog_bar=True)
        self.log("seg", L_seg, prog_bar=True)
        self.log("jac", L_jac, prog_bar=True)
        self.log("smooth", L_smooth, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vol, mask, prob_mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']

        mask_oof = (prob_mask_oof > 0.3).float()

        if self.cfg.apply_gaussian:
            mask_oof  = gaussian_blur_3d(mask_oof, self.cfg.kernel_size, self.cfg.sigma)

        if self.cfg.custom:
            x = torch.cat([vol, mask_oof, prob_mask_oof], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)


        prediction = self.sliding_window_inferer(x, self.model)
        prediction = prediction > self.cfg.threshold

        # print(prediction)
        score = calc_score(mask.cpu().numpy()[0, 0], prediction.cpu().numpy()[0, 0])
        self.val_num_samples += x.shape[0]
        self.scores.append(score.score)
        self.topo_scores.append(score.topo.toposcore)
        self.voi_scores.append(score.voi.voi_score)
        self.surface_scores.append(score.surface_dice)
        return None

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

    # def on_validation_epoch_end(self):
    #     dice = torch.cat(self.val_dice_scores).mean()
    #
    #     self.log(
    #         "val_dice",
    #         dice,
    #         prog_bar=True,
    #         rank_zero_only=True,
    #         sync_dist=True,
    #     )
    #
    #     if self.trainer.is_global_zero:
    #         print(
    #             f"\nVAL Epoch {self.current_epoch:03d} │ "
    #             f"Dice: {dice:.4f}\n"
    #         )
    #
    #     # Reset
    #     self.val_dice_scores = []
    #     self.dice_metric.reset()

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
            min_lr=5e-4,
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


class ICDiffeoRefineModule(DiffeoRefineModule):
    """
    Trainer for ICDeformDynUnet.
    Differences vs base trainer:
      - model returns (warped_mask, phis, final_phi)
      - smoothness applied to incremental deformations (optional)
      - jacobian applied to final deformation
    """

    def __init__(self, model, cfg):
        super().__init__(model, cfg)

        # IC-specific weights
        self.lambda_iter = getattr(cfg, "lambda_iter", 0.0)  # optional intermediate supervision
        self.lambda_smooth_ic = getattr(cfg, "lambda_smooth_ic", self.lambda_smooth)

    # ----------------------------------------
    # TRAINING STEP (override)
    # ----------------------------------------
    def training_step(self, batch, batch_idx):
        vol, mask, prob_mask_oof, skel = (
            batch["Image"],
            batch["Mask"],
            batch["Mask_OOF"],
            batch["Skel"],
        )

        # --- augment OOF mask ---
        threshold = torch.empty(1, device=prob_mask_oof.device).uniform_(0.1, 0.5)
        mask_oof = (prob_mask_oof > threshold).float()

        if self.cfg.apply_gaussian:
            mask_oof = gaussian_blur_3d(mask_oof, self.cfg.kernel_size, self.cfg.sigma)

        if self.cfg.custom:
            x = torch.cat([vol, mask_oof, prob_mask_oof], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)

        # ---------------- IC forward ----------------
        pred_warped, phis, final_phi = self(x, return_params=True)

        ignore_mask = mask != 2

        # ---------------- losses ----------------
        L_seg = self.seg_loss(pred_warped * ignore_mask, mask * ignore_mask)
        L_skel = self.skel_loss(pred_warped, skel, mask)
        L_topo = self.topo_loss(pred_warped * ignore_mask, mask * ignore_mask)
        L_surf = self.soft_surf_loss(pred_warped * ignore_mask, mask * ignore_mask)

        # Jacobian penalty on final deformation
        L_jac = jacobian_log_barrier(final_phi)

        # Optional smoothness on incremental fields
        L_smooth = 0.0
        if self.lambda_smooth_ic > 0 and len(phis) > 1:
            for i in range(1, len(phis)):
                dv = phis[i] - phis[i - 1]
                L_smooth += self.svf_smoothness(dv)
            L_smooth /= (len(phis) - 1)

        # Optional intermediate supervision
        L_iter = 0.0
        if self.lambda_iter > 0:
            for phi in phis[:-1]:
                warped_i = warp_vol_using_disp(mask_oof, phi)
                L_iter += self.seg_loss(warped_i * ignore_mask, mask * ignore_mask)
            L_iter /= max(len(phis) - 1, 1)

        # ---------------- total ----------------
        loss = (
            L_seg
            + 0.1 * L_skel
            + 0.5 * L_topo
            + 0.5 * L_surf
            + self.lambda_jac * L_jac
            + self.lambda_smooth_ic * L_smooth
            + self.lambda_iter * L_iter
        )

        # ---------------- logging ----------------
        self.log("loss", loss, prog_bar=True)
        self.log("seg", L_seg, prog_bar=True)
        self.log("jac", L_jac, prog_bar=True)
        self.log("smooth_ic", L_smooth, prog_bar=True)
        if self.lambda_iter > 0:
            self.log("iter_sup", L_iter, prog_bar=True)

        return loss



class DiffeoRefineModuleV2(DiffeoRefineModule):
    """
    V2:
    - model returns (corrected, v, phi, t)
    - topology losses act ONLY on corrected output
    - topology gate t is explicitly regularized
    """

    def __init__(self, model, cfg):
        super().__init__(model, cfg)

        # topology gate weights
        self.lambda_sparse = getattr(cfg, "lambda_sparse", 0.05)
        self.lambda_tv = getattr(cfg, "lambda_tv", 0.05)
        self.lambda_boundary = getattr(cfg, "lambda_boundary", 0.05)

    # -----------------------------
    # topology gate regularizers
    # -----------------------------
    def topo_sparsity(self, t):
        return t.mean()

    def topo_tv(self, t):
        return (
            (t[:,:,1:] - t[:,:,:-1]).abs().mean() +
            (t[:,:,:,1:] - t[:,:,:,:-1]).abs().mean() +
            (t[:,:,:,:,1:] - t[:,:,:,:,:-1]).abs().mean()
        ) / 3.0

    def topo_boundary(self, t, warped):
        # encourage topo edits near surface only
        boundary = warped * (1.0 - warped)
        return (t * (1.0 - boundary)).mean()

    # -----------------------------
    # TRAINING STEP (V2)
    # -----------------------------
    def training_step(self, batch, batch_idx):
        vol, mask, prob_mask_oof, skel = (
            batch["Image"],
            batch["Mask"],
            batch["Mask_OOF"],
            batch["Skel"],
        )

        # ---- OOF augmentation ----
        threshold = 0.3 + torch.randn(1, device=prob_mask_oof.device) * 0.05
        threshold = torch.clamp(threshold, 0.1, 0.5)
        mask_oof = (prob_mask_oof > threshold).float()


        if self.cfg.apply_gaussian:
            mask_oof = gaussian_blur_3d(
                mask_oof, self.cfg.kernel_size, self.cfg.sigma
            )

        if self.cfg.custom:
            x = torch.cat([vol, mask_oof, prob_mask_oof], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)

        # ---------------- forward ----------------
        corrected, v, phi, t = self(x, return_params=True)

        ignore_mask = mask != 2

        # ---------------- segmentation & topology ----------------
        L_seg = self.seg_loss(
            corrected * ignore_mask,
            mask * ignore_mask,
        )

        L_topo = self.topo_loss(
            corrected * ignore_mask,
            mask * ignore_mask,
        )

        L_skel = self.skel_loss(
            corrected,
            skel,
            mask,
        )

        L_surf = self.soft_surf_loss(
            corrected * ignore_mask,
            mask * ignore_mask,
        )

        # ---------------- diffeo regularization ----------------
        L_smooth = self.svf_smoothness(v)
        L_jac = jacobian_log_barrier(phi)

        # ---------------- topo-gate regularization ----------------
        with torch.no_grad():
            warped = warp_vol_using_disp(mask_oof, phi)

        L_sparse = self.topo_sparsity(t)
        L_tv = self.topo_tv(t)
        L_boundary = self.topo_boundary(t, warped)

        # ---------------- total loss ----------------
        loss = (
            L_seg
            + 0.5 * L_topo
            + 0.3 * L_skel
            + 0.3 * L_surf
            + self.lambda_jac * L_jac
            + self.lambda_smooth * L_smooth
            + self.lambda_sparse * L_sparse
            + self.lambda_tv * L_tv
            + self.lambda_boundary * L_boundary
        )

        # ---------------- logging ----------------
        self.log("loss", loss, prog_bar=True)
        self.log("seg", L_seg, prog_bar=True)
        self.log("topo", L_topo, prog_bar=True)
        self.log("jac", L_jac, prog_bar=True)
        self.log("smooth", L_smooth, prog_bar=True)
        self.log("t_sparse", L_sparse, prog_bar=True)
        self.log("t_tv", L_tv, prog_bar=False)
        self.log("t_boundary", L_boundary, prog_bar=False)

        return loss


class DiffeoRefineModuleV3(DiffeoRefineModule):
    """
    V3:
    - model returns (sdf_pred, prob_pred, v, phi, gate)
    - SignedDistanceLoss supervises sdf_pred
    - Dice / topo / skeleton operate on prob_pred
    """
    def __init__(self, model, cfg):
        super().__init__(model, cfg)

        # --- SDF loss ---
        self.sdf_loss = SignedDistanceLoss(cfg.sdf)

        # --- topology gate regularization ---
        self.lambda_sparse = getattr(cfg, "lambda_sparse", 0.05)
        self.lambda_tv = getattr(cfg, "lambda_tv", 0.05)
        self.lambda_boundary = getattr(cfg, "lambda_boundary", 0.05)

    # -----------------------------
    # topology gate regularizers
    # -----------------------------
    def topo_sparsity(self, t):
        return t.mean()

    def topo_tv(self, t):
        return (
            (t[:,:,1:] - t[:,:,:-1]).abs().mean() +
            (t[:,:,:,1:] - t[:,:,:,:-1]).abs().mean() +
            (t[:,:,:,:,1:] - t[:,:,:,:,:-1]).abs().mean()
        ) / 3.0

    def topo_boundary(self, t, sdf):
        # encourage edits near zero-level set
        boundary = torch.exp(-sdf.abs())
        return (t * (1.0 - boundary)).mean()

    # -----------------------------
    # TRAINING STEP (V3)
    # -----------------------------
    def training_step(self, batch, batch_idx):
        vol   = batch["Image"]
        mask  = batch["Mask"]
        sdf_gt = batch["SDF"]
        skel  = batch["Skel"]
        prob_mask_oof = batch["Mask_OOF"]

        # ---- OOF augmentation ----
        threshold = torch.empty(1, device=prob_mask_oof.device).uniform_(0.1, 0.5)
        mask_oof = (prob_mask_oof > threshold).float()

        if self.cfg.apply_gaussian:
            mask_oof = gaussian_blur_3d(
                mask_oof, self.cfg.kernel_size, self.cfg.sigma
            )
        # Convert OOF probability to SDF once
        sdf_oof = soft_sdf(mask_oof)

        x = torch.cat([vol, sdf_oof], dim=1)

        # ---------------- forward ----------------
        pred_dict = self(x, return_params=True)
        sdf_pred, prob_pred, phi, v, t  = pred_dict['sdf'], pred_dict['prob'],\
            pred_dict['phi'], pred_dict['v'], pred_dict['gate']

        ignore_mask = mask != 2

        # ---------------- SDF geometry loss ----------------
        L_sdf = self.sdf_loss(sdf_pred, sdf_gt)

        # ---------------- segmentation & topology ----------------
        L_seg = self.seg_loss(
            prob_pred * ignore_mask,
            mask * ignore_mask,
        )

        L_topo = self.topo_loss(
            prob_pred * ignore_mask,
            mask * ignore_mask,
        )

        L_skel = self.skel_loss(
            prob_pred,
            skel,
            mask,
        )

        # ---------------- diffeo regularization ----------------
        L_smooth = self.svf_smoothness(v)
        L_jac = jacobian_log_barrier(phi)

        # ---------------- topo-gate regularization ----------------
        L_sparse = self.topo_sparsity(t)
        L_tv = self.topo_tv(t)
        L_boundary = self.topo_boundary(t, sdf_pred)

        # ---------------- total loss ----------------
        loss = (
            L_sdf
            + L_seg
            + 0.5 * L_topo
            + 0.3 * L_skel
            + self.lambda_jac * L_jac
            + self.lambda_smooth * L_smooth
            + self.lambda_sparse * L_sparse
            + self.lambda_tv * L_tv
            + self.lambda_boundary * L_boundary
        )

        # ---------------- logging ----------------
        self.log("loss", loss, prog_bar=True)
        self.log("sdf", L_sdf, prog_bar=True)
        self.log("seg", L_seg, prog_bar=True)
        self.log("topo", L_topo, prog_bar=True)
        self.log("jac", L_jac, prog_bar=True)
        self.log("smooth", L_smooth, prog_bar=True)
        self.log("t_sparse", L_sparse, prog_bar=True)
        return loss