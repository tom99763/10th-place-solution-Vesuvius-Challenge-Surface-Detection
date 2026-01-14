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

        # self.seg_loss = TverskyLoss(
        #     sigmoid=False,  # ← critical fix
        #     to_onehot_y=True,  # because your labels are integer class indices
        #     softmax=False,
        #     reduction="mean",
        #     alpha=cfg.alpha,
        #     beta=cfg.beta
        # )
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
        vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']

        valid_mask = mask !=2

        # random threhsold augmentation
        if self.cfg.is_prob_oof_mask:
            threshold = 0.3
            mask_oof = (mask_oof > threshold).float()

        if self.cfg.apply_gaussian:
            mask_oof = gaussian_blur_3d(mask_oof, self.cfg.kernel_size, self.cfg.sigma)

        x = torch.cat([vol, mask_oof], dim=1)

        # Sliding window inference
        logits = self.sliding_window_inferer(x, self.model)

        # Binarize
        pred = (logits > self.cfg.threshold).float()

        # Ensure shapes: (B, C, D, H, W)
        if pred.ndim == 4:
            pred = pred.unsqueeze(1)
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)

        # Dice
        dice = self.dice_metric(pred * valid_mask, mask * valid_mask)
        self.val_dice_scores.append(dice)

        return None

    # def validation_step(self, batch, batch_idx):
    #     vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']
    #     if self.cfg.apply_gaussian:
    #         x = torch.cat([vol, gaussian_blur_3d(mask_oof, self.cfg.kernel_size, self.cfg.sigma)], dim=1)
    #     else:
    #         x = torch.cat([vol, mask_oof], dim=1)
    #
    #     # if self.current_epoch != 0:
    #     #     prediction = self.sliding_window_inferer(x, self.model)
    #     #     prediction = prediction > self.cfg.threshold
    #     # else:
    #     #     prediction = mask_oof
    #
    #     prediction = self.sliding_window_inferer(x, self.model)
    #     prediction = prediction > self.cfg.threshold
    #
    #     # print(prediction)
    #     score = calc_score(mask.cpu().numpy()[0, 0], prediction.cpu().numpy()[0, 0])
    #     self.val_num_samples += x.shape[0]
    #     self.scores.append(score.score)
    #     self.topo_scores.append(score.topo.toposcore)
    #     self.voi_scores.append(score.voi.voi_score)
    #     self.surface_scores.append(score.surface_dice)
    #     return None

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
        dice = torch.cat(self.val_dice_scores).mean()

        self.log(
            "val_dice",
            dice,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        if self.trainer.is_global_zero:
            print(
                f"\nVAL Epoch {self.current_epoch:03d} │ "
                f"Dice: {dice:.4f}\n"
            )

        # Reset
        self.val_dice_scores = []
        self.dice_metric.reset()

    # def on_validation_epoch_end(self):
    #     comp_score = np.stack(self.scores).sum() / self.val_num_samples
    #     topo_score = np.stack(self.topo_scores).sum() / self.val_num_samples
    #     voi_score = np.stack(self.voi_scores).sum() / self.val_num_samples
    #     surface_score = np.stack(self.surface_scores).sum() / self.val_num_samples
    #
    #     # === Logging ===
    #     self.log("val_topo", topo_score, prog_bar=True, rank_zero_only=True)
    #     self.log("val_voi", voi_score, prog_bar=True, rank_zero_only=True)
    #     self.log("val_surface", surface_score, prog_bar=True, rank_zero_only=True)
    #     self.log("val_comp_metric", comp_score, prog_bar=True, rank_zero_only=True, sync_dist=True)
    #
    #     if self.trainer.is_global_zero:
    #         print(f"\nVAL Epoch {self.current_epoch:03d} │ "
    #               f"VOI: {voi_score:.4f} │ "
    #               f"Topo: {topo_score:.4f} │ "
    #               f"Surf: {surface_score:.4f} │ "
    #               f"→ COMP: {comp_score:.4f} ←\n")
    #
    #     # === Reset everything ===
    #     self.scores = []
    #     self.topo_scores = []
    #     self.voi_scores = []
    #     self.surface_scores = []
    #     self.val_num_samples = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-2,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            cooldown=2,
            min_lr=1e-4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_dice",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# --------------------
# --------------------
# Inverse Compositional Deformation Network 3D
# --------------------
# --------------------
class ICDeformDynUnet(DeformDynUnet):
    """
    Iterative inverse-compositional deformable warper for 3D masks.
    Predictor: shared network that predicts a small SVF (B,3,D,H,W) given [vol, warped_mask]
    At each iteration:
      - delta_v = predictor([vol, warped_mask])  (small SVF)
      - delta_phi = exp(delta_v)  (scaling-and-squaring)
      - inv_delta_phi = exp(-delta_v)  (exact inverse in SVF group)
      - phi <- inv_delta_phi + warp(phi, inv_delta_phi)  (IC composition: phi <- phi ∘ Δφ^{-1})
      - warped_mask = warp(orig_mask, phi)
    Notes:
      - We always sample the mask from the original template (deferred warping).
      - warp_displacement(field, by_disp) warps `field` by `by_disp`.
    """
    def __init__(self, cfg, predictor=None):
        super().__init__(cfg)
        # predictor already set by parent (instantiate(cfg.models))
        self.num_iters = getattr(cfg, "num_iters", 3)
        self.max_delta_v = getattr(cfg, "max_delta_v", cfg.max_v if hasattr(cfg, "max_v") else 1.0)
        self.n_steps = getattr(cfg, "n_steps", cfg.n_steps if hasattr(cfg, "n_steps") else 6)

    def forward(self, x, return_params=False):
        """
        x is concatenation of vol and mask: B, 2, D, H, W
        returns warped_mask; if return_params=True also returns list of phis and final phi
        """
        vol, mask = x[:, 0:1], x[:, 1:2]
        B, _, D, H, W = mask.shape
        device = mask.device

        # initialize phi = zero displacement (identity)
        phi = torch.zeros(B, 3, D, H, W, device=device, dtype=mask.dtype)

        orig_mask = mask
        warped_mask = orig_mask  # initial

        phis = []

        for it in range(self.num_iters):
            # predictor input: volume and current warped mask
            inp = torch.cat([vol, warped_mask], dim=1)  # B, C_img + C_mask, D,H,W
            raw_delta_v = self.predictor(inp)  # expected B,3,D,H,W

            if raw_delta_v.shape[1] != 3:
                raise RuntimeError(f"predictor must output 3 channels for voxel SVF, got {raw_delta_v.shape}")

            # small incremental SVF
            delta_v = torch.tanh(raw_delta_v) * self.max_delta_v  # keep small

            # exponentiate to delta_phi and inverse via -delta_v
            # Δφ = exp(Δv); Δφ^{-1} = exp(-Δv) exactly in SVF paramization
            # We compute only inv_delta_phi explicitly (that's what IC uses)
            inv_delta_phi = scaling_and_squaring(-delta_v, n_steps=self.n_steps)

            # INVERSE-COMPOSITION (left composition by inv_delta_phi):
            # phi_new = inv_delta_phi + warp(phi, inv_delta_phi)
            warped_phi = warp_displacement(phi, inv_delta_phi)  # sample phi at positions after inv_delta_phi
            phi = inv_delta_phi + warped_phi

            # update the warped mask by applying updated phi to the original mask (deferred warping)
            warped_mask = warp_vol_using_disp(orig_mask, phi)

            phis.append(phi)

        if return_params:
            return warped_mask, phis, phi
        return warped_mask


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
        vol, mask, mask_oof, skel = (
            batch["Image"],
            batch["Mask"],
            batch["Mask_OOF"],
            batch["Skel"],
        )

        # --- augment OOF mask ---
        if self.cfg.is_prob_oof_mask:
            threshold = torch.empty(1, device=mask_oof.device).uniform_(0.1, 0.5)
            mask_oof = (mask_oof > threshold).float()

        if self.cfg.apply_gaussian:
            mask_oof = gaussian_blur_3d(mask_oof, self.cfg.kernel_size, self.cfg.sigma)

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

