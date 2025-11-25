from monai.inferers.inferer import SlidingWindowInfererAdapt
from monai.metrics import DiceMetric
import pytorch_lightning as pl
from src.trainers.losses import *
from monai.losses import DiceCELoss

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

        # Loss
        self.seg_loss = DiceCELoss(
            sigmoid=True,  # ← critical fix
            to_onehot_y=True,  # because your labels are integer class indices
            softmax=False,
            reduction="mean",
            squared_pred=True,
            lambda_ce=cfg.lambda_ce,
            lambda_dice=cfg.lambda_dice,
        )

        # Metric
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            ignore_empty=True
        )
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=cfg.input_size, sw_batch_size=4, overlap=0, mode="constant"
        )

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
        x = torch.cat([vol, mask_oof], dim=1)
        pred_warped, v, phi = self(x, return_params=True)
        ignore_mask = mask != 2

        # segmentation loss using MONAI DiceCE
        L_seg = self.seg_loss(pred_warped * ignore_mask, mask * ignore_mask)

        # smoothness regularizer
        L_smooth = self.svf_smoothness(v)

        # jacobian folding penalty
        # det = jacobian_determinant(phi)
        # L_jac = torch.relu(-det).mean()
        L_jac = jacobian_log_barrier(phi)

        loss = L_seg + self.lambda_smooth * L_smooth + self.lambda_jac * L_jac

        self.log("loss", loss, prog_bar=True)
        self.log("seg", L_seg)
        self.log("jac", L_jac)
        self.log("smooth", L_smooth)
        return loss

    # ----------------------------------------
    # VALIDATION STEP
    # ----------------------------------------
    def validation_step(self, batch, batch_idx):
        vol, mask, mask_oof = batch['Image'], batch['Mask'], batch['Mask_OOF']
        x = torch.cat([vol, mask_oof], dim=1)
        ignore_mask = mask != 2
        pred_logits = self.sliding_window_inferer(x, self.model)
        self.dice_metric(y_pred=(pred_logits * ignore_mask).clamp(min=0.0, max=1.0), y = mask * ignore_mask)

    def on_validation_epoch_end(self):
        dice_score = self.dice_metric.aggregate().mean().item()
        self.log("val_dice", dice_score, prog_bar=True)
        self.dice_metric.reset()

    # ----------------------------------------
    # Optimizer
    # ----------------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
