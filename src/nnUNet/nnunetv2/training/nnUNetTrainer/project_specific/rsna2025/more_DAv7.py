from typing import Union, Tuple, List

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.rot90 import Rot90Transform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.spatial.transpose import TransposeAxesTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from .more_DAv3 import RSNA2025Trainer_moreDAv3
from .more_DAv6 import SimulateThickSliceTransform


class RSNA2025Trainer_moreDAv7(RSNA2025Trainer_moreDAv3):
    """v7: v6系を基に全軸ミラーと対称軸系ランダム変換を追加したTrainer"""

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """全ての空間軸に対するミラーリングを有効化"""
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, _ = (
            nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size(self)
        )
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        mirror_axes = tuple(range(dim)) if dim > 0 else None
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

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

        matching_axes = np.array([sum(i == j for j in patch_size) for i in patch_size])
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
        assert len(valid_axes) == len(patch_size), f"valid_axes: {valid_axes} != patch_size: {patch_size}"

        if valid_axes and np.any(matching_axes > 1):
            allowed_axes = set(int(ax) for ax in valid_axes)
            additional_transforms = [
                RandomTransform(
                    Rot90Transform(
                        num_rot_per_combination=(1, 2, 3),
                        num_axis_combinations=(1, 4),
                        allowed_axes=allowed_axes,
                    ),
                    apply_probability=0.5,
                ),
                RandomTransform(
                    TransposeAxesTransform(allowed_axes=allowed_axes),
                    apply_probability=0.5,
                ),
            ]
            mirror_indices = [
                i
                for i, transform in enumerate(base_transforms.transforms)
                if isinstance(transform, MirrorTransform)
            ]
            insert_position = mirror_indices[0] if mirror_indices else len(base_transforms.transforms)
            for offset, extra_transform in enumerate(additional_transforms):
                base_transforms.transforms.insert(insert_position + offset, extra_transform)

        return base_transforms
