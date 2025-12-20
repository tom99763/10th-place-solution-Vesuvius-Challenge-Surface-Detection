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
            roi_size=cfg.input_size, sw_batch_size=2, overlap=0.5, mode="gaussian"
        )

        # === Topology & surface ===
        self.cldice = FastClDiceLoss()
        self.sdf = SoftSDFLoss(tau_vox=2.0)
        self.surface = SurfaceLoss(tau_vox=2.0)

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