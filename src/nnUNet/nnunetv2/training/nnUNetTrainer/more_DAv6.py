from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import warnings
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.tversky_loss import MemoryEfficientTverskyLoss
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.skeleton_recall_loss import (
    DC_SkelREC_and_CE_loss,
)

from .more_DAv3 import RSNA2025Trainer_moreDAv3
from .nnUNetTrainerSkeletonRecall_more_DAv3 import nnUNetTrainerSkeletonRecall_more_DAv3
from nnunetv2.training.data_augmentation.custom_transforms.skeletonization import SkeletonTransform
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform


class SimulateThickSliceTransform(BasicTransform):
    """疑似的にスライス厚を増やすために一方向にダウンサンプリングしてから元解像度に戻すTransform"""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.25, 0.6),
        candidate_axes: Tuple[int, ...] = (0, 1, 2),
    ) -> None:
        super().__init__()
        self.scale_range = scale_range
        self.candidate_axes = candidate_axes

    @staticmethod
    def _resize_tensor(tensor: torch.Tensor, target_size: Tuple[int, ...]) -> torch.Tensor:
        # 常にfloatへ変換して補間し、必要なら元dtypeに戻す
        original_dtype = tensor.dtype
        needs_cast = not tensor.is_floating_point()
        work_tensor = tensor.float() if needs_cast else tensor
        resized = F.interpolate(work_tensor.unsqueeze(0), size=target_size, mode="nearest")
        resized = resized.squeeze(0)
        if needs_cast:
            resized = resized.to(original_dtype)
        return resized

    def __call__(self, **data_dict):
        image = data_dict.get("image")
        if image is None:
            return data_dict

        if image.ndim != 4:
            return data_dict

        spatial_shape = image.shape[1:]
        valid_axes = [ax for ax in self.candidate_axes if 0 <= ax < len(spatial_shape)]
        if not valid_axes:
            return data_dict

        chosen_axis = int(np.random.choice(valid_axes))
        scale = float(np.random.uniform(*self.scale_range))
        target_shape = list(spatial_shape)
        target_dim = max(1, int(round(target_shape[chosen_axis] * scale)))

        if target_dim == target_shape[chosen_axis]:
            return data_dict

        target_shape[chosen_axis] = target_dim
        target_shape_tuple = tuple(target_shape)

        img_is_numpy = isinstance(image, np.ndarray)
        img_tensor = torch.from_numpy(image) if img_is_numpy else image
        img_original_dtype = img_tensor.dtype
        img_tensor = img_tensor.to(dtype=torch.float32)

        downsampled = self._resize_tensor(img_tensor, target_shape_tuple)
        restored = F.interpolate(downsampled.unsqueeze(0), size=spatial_shape, mode="nearest").squeeze(0)

        if img_is_numpy:
            restored_np = restored.cpu().numpy()
            if np.issubdtype(image.dtype, np.integer):
                restored_np = np.rint(restored_np)
            data_dict["image"] = restored_np.astype(image.dtype)
        else:
            if img_original_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                restored = torch.round(restored).to(img_original_dtype)
            else:
                restored = restored.to(img_original_dtype)
            data_dict["image"] = restored

        segmentation = data_dict.get("segmentation")
        if segmentation is None:
            return data_dict

        seg_is_numpy = isinstance(segmentation, np.ndarray)
        seg_tensor = torch.from_numpy(segmentation) if seg_is_numpy else segmentation
        seg_original_dtype = seg_tensor.dtype
        seg_tensor = seg_tensor.to(dtype=torch.float32)

        seg_downsampled = self._resize_tensor(seg_tensor, target_shape_tuple)
        seg_restored = F.interpolate(
            seg_downsampled.unsqueeze(0), size=spatial_shape, mode="nearest"
        ).squeeze(0)

        if seg_original_dtype == torch.bool:
            seg_restored = seg_restored > 0.5
        elif seg_original_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            seg_restored = torch.round(seg_restored).to(seg_original_dtype)
        else:
            seg_restored = seg_restored.to(seg_original_dtype)

        if seg_is_numpy:
            seg_np = seg_restored.cpu().numpy()
            if np.issubdtype(segmentation.dtype, np.integer):
                seg_np = np.rint(seg_np)
            data_dict["segmentation"] = seg_np.astype(segmentation.dtype)
        else:
            data_dict["segmentation"] = seg_restored

        return data_dict


class RSNA2025Trainer_moreDAv6(RSNA2025Trainer_moreDAv3):
    """v6: v3系に疑似スライス厚拡張を追加したTrainer"""

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        base_transforms = super().get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )

        thick_slice_transform = RandomTransform(
            SimulateThickSliceTransform(scale_range=(0.05, 0.6), candidate_axes=(0, 1, 2)),
            apply_probability=0.4,
        )

        insert_idx = None
        for idx, transform in enumerate(base_transforms.transforms):
            if isinstance(transform, SpatialTransform):
                insert_idx = idx
                break

        if insert_idx is None:
            base_transforms.transforms.append(thick_slice_transform)
        else:
            base_transforms.transforms.insert(insert_idx, thick_slice_transform)

        return base_transforms


class RSNA2025Trainer_moreDAv6_SkeletonRecall(nnUNetTrainerSkeletonRecall_more_DAv3):
    """SkeletonRecallロスを組み込んだv6トレーナー"""

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        base_transforms = super().get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )

        thick_slice_transform = RandomTransform(
            SimulateThickSliceTransform(scale_range=(0.05, 0.6), candidate_axes=(0, 1, 2)),
            apply_probability=0.4,
        )

        insert_idx = None
        for idx, transform in enumerate(base_transforms.transforms):
            if isinstance(transform, SpatialTransform):
                insert_idx = idx
                break

        if insert_idx is None:
            base_transforms.transforms.append(thick_slice_transform)
        else:
            base_transforms.transforms.insert(insert_idx, thick_slice_transform)

        return base_transforms


# v6.1: v6系に疑似スライス厚拡張を0.3~0.6にしたTrainer
class RSNA2025Trainer_moreDAv6_1_SkeletonRecall(nnUNetTrainerSkeletonRecall_more_DAv3):
    """SkeletonRecallロスを組み込んだv6トレーナー"""

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        base_transforms = super().get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )

        thick_slice_transform = RandomTransform(
            SimulateThickSliceTransform(scale_range=(0.3, 0.6), candidate_axes=(0, 1, 2)),
            apply_probability=0.4,
        )

        insert_idx = None
        for idx, transform in enumerate(base_transforms.transforms):
            if isinstance(transform, SpatialTransform):
                insert_idx = idx
                break

        if insert_idx is None:
            base_transforms.transforms.append(thick_slice_transform)
        else:
            base_transforms.transforms.insert(insert_idx, thick_slice_transform)

        return base_transforms


class RSNA2025Trainer_moreDAv6_SkeletonRecallTverskyBeta07(RSNA2025Trainer_moreDAv6_SkeletonRecall):
    """Diceの代わりにTversky Loss (β=0.7) を用いるSkeletonRecallトレーナー"""

    def _build_loss(self):
        """Tversky Lossを用いたSkeletonRecall向け複合損失を構築"""
        if self.label_manager.ignore_label is not None:
            warnings.warn(
                "Support for ignore label with Skeleton Recall is experimental and may not work as expected"
            )

        loss = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
                "alpha": 0.3,
                "beta": 0.7,
            },
            soft_skelrec_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=1,
            weight_dice=1,
            weight_srec=self.weight_srec,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientTverskyLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07(RSNA2025Trainer_moreDAv6_1_SkeletonRecall):
    """Diceの代わりにTversky Loss (β=0.7) を用いるSkeletonRecallトレーナー"""

    def _build_loss(self):
        """Tversky Lossを用いたSkeletonRecall向け複合損失を構築"""
        if self.label_manager.ignore_label is not None:
            warnings.warn(
                "Support for ignore label with Skeleton Recall is experimental and may not work as expected"
            )

        loss = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
                "alpha": 0.3,
                "beta": 0.7,
            },
            soft_skelrec_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=1,
            weight_dice=1,
            weight_srec=self.weight_srec,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientTverskyLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07(
    RSNA2025Trainer_moreDAv6_SkeletonRecallTverskyBeta07
):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_srec = 3


class RSNA2025Trainer_moreDAv6_SkeletonRecallW3(RSNA2025Trainer_moreDAv6_SkeletonRecall):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_srec = 3


class RSNA2025Trainer_moreDAv6_SkeletonRecallW5(RSNA2025Trainer_moreDAv6_SkeletonRecall):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_srec = 5
