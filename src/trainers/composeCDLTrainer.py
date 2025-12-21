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
class ComposeCDLRefineModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg.lr
        self.D = torch.from_numpy(load_sparse_tensor(str(Path(self.cfg.dictionary_path) / 'dictionary.npz')))

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

    def forward(self, x):
        return self.model(x)

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        vol, Z_gt = batch["Image"], batch["Z"]  # Z_gt: (B,K,D,H,W)
        Z_hat = self.model(vol)

        # Loss: MSE between predicted and target Z
        loss = F.mse_loss(Z_hat, Z_gt)

        # Optional sparsity regularization
        if getattr(self.cfg, "lambda_sparse", 0) > 0:
            loss += self.cfg.lambda_sparse * Z_hat.abs().mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        vol, mask = batch['Image'], batch['Mask']
        if self.current_epoch != 0:
            prediction_Z = self.sliding_window_inferer(vol, self.model)
            D = self.D.to(prediction_Z.device)
            prediction = reconstruct_mask(prediction_Z, D)
        else:
            prediction = mask

        score = calc_score(mask.cpu().numpy()[0, 0] ,prediction.cpu().numpy()[0, 0])
        self.val_num_samples += vol.shape[0]
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