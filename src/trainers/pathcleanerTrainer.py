from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
import pytorch_lightning as pl
from src.trainers.losses import *
from monai.losses import DiceCELoss
from src.procs.proc_data import *
import sys
sys.path.append('../../')
from cv import calc_score

# ---------------------
# Lightning Module
# ---------------------
class CleanerRefineModule(pl.LightningModule):
    def __init__(self, model, cfg ):
        super().__init__()
        self.model = model
        self.lr = cfg.lr
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

        # For proper epoch-level averaging
        self.val_num_samples = 0

    def forward(self, x):
        return self.model(x)

    # ----------------------------------------
    # TRAINING STEP
    # ----------------------------------------
    def training_step(self, batch, batch_idx):
        vol, mask, mask_oof, invalid_mask = batch['Image'], batch['Mask'], batch['Mask_OOF'], batch['Invalid_Mask']
        x = torch.cat([vol, mask_oof], dim=1)
        pred_prob = self(x)
        ignore_mask = mask != 2

        # segmentation loss using MONAI DiceCE
        L_seg = self.seg_loss(pred_prob * ignore_mask, invalid_mask * ignore_mask)
        L_topo = self.topo_loss(pred_prob * ignore_mask, invalid_mask * ignore_mask)
        L_surf = self.soft_surf_loss(pred_prob * ignore_mask, invalid_mask * ignore_mask)

        loss = L_seg + 0.5 * L_topo + 0.5 * L_surf

        self.log("loss", loss, prog_bar=True)
        self.log("seg", L_seg, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vol, mask, mask_oof, invalid_mask = batch['Image'], batch['Mask'], batch['Mask_OOF'], batch['Invalid_Mask']
        x = torch.cat([vol, mask_oof], dim=1)

        prediction = self.sliding_window_inferer(x, self.model)
        prediction = F.relu(mask_oof - prediction) > self.cfg.threshold

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