from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
import pytorch_lightning as pl
from networkx.classes import edges

from src.trainers.losses import *
from monai.losses import DiceCELoss
from src.procs.proc_data import *
import torch.nn as nn
import sys
sys.path.append('../../')
from cv import calc_score

# ---------------------
# Lightning Module
# ---------------------
class ComposeRefineModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg.lr

        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=cfg.input_size, sw_batch_size=2, overlap=0.5, mode="gaussian"
        )

        # Validation accumulators (scalar averages)
        self.scores = []
        self.topo_scores = []
        self.voi_scores = []
        self.surface_scores =[]

        # For proper epoch-level averaging
        self.val_num_samples = 0

    def forward(self, x, return_components=False):
        return self.model(x, return_components=return_components)

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        vol, mask_oof, mask = batch["Image"], batch["Mask_OOF"], batch["Mask"]
        sdf_gt = batch["C1"]
        normals_gt = batch["C2"]
        thickness_gt = batch["C3"]
        mask_tensor = (mask != 2).float()  # valid mask for loss

        # --- Optional Gaussian preprocessing ---
        if self.cfg.apply_gaussian:
            x = torch.cat([vol,
                           gaussian_blur_3d(mask_oof, self.cfg.kernel_size, self.cfg.sigma)], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)

        # --- Forward pass ---
        comps = self(x, return_components=True)
        pred_sdf = comps["sdf"]
        pred_normals = comps["normals"]
        pred_thickness = comps["thickness"]

        # --- Losses ---
        sdf_loss = F.smooth_l1_loss(pred_sdf * mask_tensor, sdf_gt * mask_tensor)
        thickness_loss = F.smooth_l1_loss(pred_thickness * mask_tensor, thickness_gt * mask_tensor)

        # Normalize normals to unit vectors before loss
        pred_normals_unit = F.normalize(pred_normals, dim=1)
        normals_gt_unit = F.normalize(normals_gt, dim=1)
        normals_loss = F.mse_loss(pred_normals_unit * mask_tensor, normals_gt_unit * mask_tensor)

        # Total loss (weighted)
        loss = self.cfg.lambda_sdf * sdf_loss + \
               self.cfg.lambda_thickness * thickness_loss + \
               self.cfg.lambda_normals * normals_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_sdf_loss", sdf_loss, prog_bar=False)
        self.log("train_thickness_loss", thickness_loss, prog_bar=False)
        self.log("train_normals_loss", normals_loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']
        if self.cfg.apply_gaussian:
            x = torch.cat([vol, gaussian_blur_3d(mask_oof)], dim=1)
        else:
            x = torch.cat([vol, mask_oof], dim=1)

        if self.current_epoch != 0:
            prediction = self.sliding_window_inferer(x, self.model)
            prediction = prediction > self.cfg.threshold
        else:
            prediction = mask_oof
        score = calc_score(mask.cpu().numpy()[0, 0] ,prediction.cpu().numpy()[0, 0])
        self.val_num_samples += x.shape[0]
        self.scores.append(score.score)
        self.topo_scores.append(score.topo.toposcore)
        self.voi_scores.append(score.voi.voi_score)
        self.surface_scores.append(score.surface_dice)
        return None


    def on_validation_epoch_end(self):
        comp_score = np.stack(self.scores).sum() / self.val_num_samples
        topo_score = np.stack(self.topo_scores).sum() / self.val_num_samples
        voi_score = np.stack(self.voi_scores).sum() / self.val_num_samples
        surface_score = np.stack(self.surface_scores).sum() / self.val_num_samples

        # === Logging ===
        self.log("val_topo", topo_score, prog_bar=True, rank_zero_only=True)
        self.log("val_voi", voi_score, prog_bar=True, rank_zero_only=True)
        self.log("val_surface", surface_score, prog_bar=True, rank_zero_only=True)
        self.log("val_comp_metric", comp_score, prog_bar=True, rank_zero_only=True, sync_dist=True)

        if self.trainer.is_global_zero:
            print(f"\nVAL Epoch {self.current_epoch:03d} │ "
                  f"VOI: {voi_score:.4f} │ "
                  f"Topo: {topo_score:.4f} │ "
                  f"Surf: {surface_score:.4f} │ "
                  f"→ COMP: {comp_score:.4f} ←\n")

        # === Reset everything ===
        self.scores = []
        self.topo_scores = []
        self.voi_scores = []
        self.surface_scores = []
        self.val_num_samples = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-2
        )