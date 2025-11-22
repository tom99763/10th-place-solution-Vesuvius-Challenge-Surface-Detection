import numpy as np
import torch
from torch import autocast
from typing import Tuple, Union, List
import warnings
from nnunetv2.training.nnUNetTrainer.approx_betti_matching_loss import BettiDicCELosss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform


class nnUNetTrainerBettiMatching(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.label_manager.has_regions:
            raise NotImplementedError("trainer not implemented for regions")

    def _build_loss(self):
        if self.label_manager.ignore_label is not None:
            warnings.warn(
                "Support for ignore label with Skeleton Recall is experimental and may not work as expected"
            )
        loss = BettiDicCELosss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            betti_kwargs={
                'eps': 0.05,
                'resolution': 400,
                'target_class': 1,
                'topo_weight': 20.0,
                'sharpness': 100,  # Threshold sharpness (higher = harder binary)
            },
            ce_kwargs={},
            weight_ce=1,
            weight_dice=1,
            weight_betti=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerBettiMatching_onlyMirror01(nnUNetTrainerBettiMatching):
    """
    Only mirrors along spatial axes 0 and 1 for 3D and 0 for 2D
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            mirror_axes = (0,)
        else:
            mirror_axes = (0, 1)
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)


class nnUNetTrainerBettiMatching_NoMirror(nnUNetTrainerNoMirroring, nnUNetTrainerBettiMatching):
    """`nnUNetTrainerSkeletonRecall` のミラーリング無効版"""
