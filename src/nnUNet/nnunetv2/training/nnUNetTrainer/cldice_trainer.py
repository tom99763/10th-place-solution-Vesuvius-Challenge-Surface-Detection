import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss




import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_FastClDice_and_CE_loss(nn.Module):
    """
    Dice + CE + FastClDice loss.
    """
    def __init__(
        self,
        weight_dice=1.0,
        weight_ce=1.0,
        weight_cldice=2.0,
        smooth=1e-5,
        dice_class=MemoryEfficientSoftDiceLoss,
        ce_kwargs=None,
        dice_kwargs=None,
        ignore_label=None,
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.smooth = smooth
        self.ignore_label = ignore_label

        ce_kwargs = ce_kwargs or {}
        dice_kwargs = dice_kwargs or {}

        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be (B, 1, D, H, W) with class indices; ignore_label is supported.
        """
        # --- Handle ignore label exactly like DC_SkelREC_and_CE_loss ---
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "ignore_label not implemented for one-hot encoded targets"
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        # --- Dice loss ---
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0

        # --- CE loss ---
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        # --- FastClDice ---
        pred_sigmoid = torch.sigmoid(net_output).clamp(0.0, 1.0)
        target_float = target_dice.float()

        if mask is not None:
            # expand mask to match channel dimension
            mask_float = mask.float()
            target_float = target_float * mask_float
            pred_sigmoid = pred_sigmoid * mask_float

        pad = (1, 1, 1, 1, 1, 1)
        pred_padded = F.pad(pred_sigmoid, pad, mode='replicate')
        local_avg = F.avg_pool3d(pred_padded, kernel_size=3, stride=1, padding=0)
        weight_pred = torch.abs(pred_sigmoid - local_avg)

        target_padded = F.pad(target_float, pad, mode='constant', value=0)
        local_avg_gt = F.avg_pool3d(target_padded, kernel_size=3, stride=1, padding=0)
        weight_true = torch.abs(target_float - local_avg_gt)

        tprec = (weight_pred * target_float).sum() / (weight_pred.sum() + self.smooth)
        tsens = (weight_true * pred_sigmoid).sum() / (weight_true.sum() + self.smooth)
        cldice_loss = 1.0 - 2.0 * tprec * tsens / (tprec + tsens + self.smooth)

        # --- Combine all ---
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cldice * cldice_loss
        return result



class nnUNetTrainerFastClDice(nnUNetTrainer):
    def _build_loss(self):
        # Create combined Dice+BCE + FastClDice loss
        loss = DC_FastClDice_and_CE_loss(weight_dice=0.8, weight_ce=0.5, weight_cldice=2.0,ignore_label=self.label_manager.ignore_label)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # Exponentially decreasing weights
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0  # ignore lowest resolution
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss