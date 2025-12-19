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
        self.l1_loss = nn.SmoothL1Loss(beta=0.1)
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

        self.cldice = FastClDiceLoss()  # ← main topology driver
        self.surface = SurfaceLoss(tau_vox=2.0)  # ← main SurfaceDice driver

        # Validation accumulators (scalar averages)
        self.scores = []
        self.topo_scores = []
        self.voi_scores = []
        self.surface_scores =[]

        # For proper epoch-level averaging
        self.val_num_samples = 0

    def forward(self, x):
        return self.model(x)

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        vol, mask_oof, mask = batch["Image"], batch["Mask_OOF"], batch["Mask"]
        ignore_mask = (mask != 2).float()
        x = torch.cat([vol, mask_oof], dim=1)
        mask_pred, residual_pred = self(x)
        loss = self.l1_loss(residual_pred * ignore_mask, (mask - mask_oof) * ignore_mask) +\
            self.seg_loss(mask_pred * ignore_mask, mask * ignore_mask)
        self.log_dict({
            "loss": loss,
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']
        x = torch.cat([vol, mask_oof], dim=1)

        prediction = self.sliding_window_inferer(x, self.model)
        prediction = prediction.clamp(0, 1) > self.cfg.threshold

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