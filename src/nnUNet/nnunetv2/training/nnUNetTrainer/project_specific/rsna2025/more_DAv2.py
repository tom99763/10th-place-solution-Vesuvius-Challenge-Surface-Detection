# ref: https://github.com/MIC-DKFZ/kaggle_BYU_Locating_Bacterial-Flagellar_Motors_2025_solution/tree/eafb1dfefccba71d629a64fc6619207d25197c42
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
    BrightnessAdditiveTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.intensity.inversion import InvertImageTransform
from batchgeneratorsv2.transforms.intensity.random_clip import CutOffOutliersTransform
from batchgeneratorsv2.transforms.local.brightness_gradient import BrightnessGradientAdditiveTransform
from batchgeneratorsv2.transforms.local.local_gamma import LocalGammaTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.noise.median_filter import MedianFilterTransform
from batchgeneratorsv2.transforms.noise.sharpen import SharpeningTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.rot90 import Rot90Transform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.spatial.transpose import TransposeAxesTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform, OneOfTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size

# from nnunetv2.training.data_augmentation.kaggle_byu_motor_regression import ConvertSegToRegrTarget
# from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.losses.bce_topk import \
#     MotorRegressionTrainer_BCEtopK20Loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainer_onlyMirror01,
)
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    _brightnessadditive_localgamma_transform_scale,
    _brightness_gradient_additive_max_strength,
    _local_gamma_gamma,
)


# v2
# 90度回転、軸転置を削除
# メディアンフィルタ、ガウシアンブラーを追加
# SimulateLowResolutionTransformの確率をデフォルトと同じ0.25に変更
class RSNA2025Trainer_moreDAv2(nnUNetTrainer_onlyMirror01):
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
        matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
        transforms = []

        transforms.append(
            RandomTransform(
                CutOffOutliersTransform(
                    (0, 2.5), (98.5, 100), p_synchronize_channels=1, p_per_channel=0.5, p_retain_std=0.5
                ),
                apply_probability=0.2,
            )
        )

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.3,
                rotation=rotation_for_DA,
                p_scaling=0.3,
                scaling=(0.6, 1.67),
                p_synchronize_scaling_across_axes=0.8,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # 90度回転、軸転置は不適切？
        # if np.any(matching_axes > 1):
        #     transforms.append(
        #         RandomTransform(
        #             Rot90Transform(
        #                 num_rot_per_combination=(1, 2, 3),
        #                 num_axis_combinations=(1, 4),
        #                 allowed_axes=set(valid_axes),
        #             ),
        #             apply_probability=0.5,
        #         )
        #     )
        #     transforms.append(
        #         RandomTransform(TransposeAxesTransform(allowed_axes=set(valid_axes)), apply_probability=0.5)
        #     )

        transforms.append(
            OneOfTransform(
                [
                    RandomTransform(
                        MedianFilterTransform((2, 8), p_same_for_each_channel=0.5, p_per_channel=0.5),
                        apply_probability=0.2,
                    ),
                    RandomTransform(
                        GaussianBlurTransform(
                            blur_sigma=(0.3, 1.5),
                            synchronize_channels=False,
                            synchronize_axes=False,
                            p_per_channel=0.5,
                            benchmark=True,
                        ),
                        apply_probability=0.2,
                    ),
                ]
            )
        )

        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(noise_variance=(0, 0.2), p_per_channel=0.5, synchronize_channels=True),
                apply_probability=0.3,
            )
        )

        transforms.append(
            RandomTransform(
                BrightnessAdditiveTransform(0, 0.5, per_channel=True, p_per_channel=0.5),
                apply_probability=0.1,
            )
        )

        transforms.append(
            OneOfTransform(
                [
                    RandomTransform(
                        ContrastTransform(
                            contrast_range=BGContrast((0.75, 1.25)),
                            preserve_range=True,
                            synchronize_channels=False,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        MultiplicativeBrightnessTransform(
                            multiplier_range=BGContrast((0.75, 1.25)),
                            synchronize_channels=False,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.3,
                    ),
                ]
            )
        )

        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=0.25,
            )
        )

        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.6, 2)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.2,
            )
        )

        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.6, 2)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.2,
            )
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        # transforms.append(RandomTransform(
        #     BlankRectangleTransform(
        #         [[max(1, p // 10), p // 3] for p in patch_size],
        #         rectangle_value=torch.mean,
        #         num_rectangles=(1, 5),
        #         force_square=False,
        #         p_per_channel=0.5
        #     ),
        #     apply_probability=0.2)
        # )

        transforms.append(
            RandomTransform(
                BrightnessGradientAdditiveTransform(
                    _brightnessadditive_localgamma_transform_scale,
                    (-0.5, 1.5),
                    max_strength=_brightness_gradient_additive_max_strength,
                    same_for_all_channels=False,
                    mean_centered=True,
                    clip_intensities=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2,
            )
        )

        transforms.append(
            RandomTransform(
                LocalGammaTransform(
                    _brightnessadditive_localgamma_transform_scale,
                    (-0.5, 1.5),
                    _local_gamma_gamma,
                    same_for_all_channels=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2,
            )
        )

        transforms.append(
            RandomTransform(
                SharpeningTransform(
                    (0.1, 1.5), p_same_for_each_channel=0.5, p_per_channel=0.5, p_clamp_intensities=0.5
                ),
                apply_probability=0.2,
            )
        )

        transforms.append(
            RandomTransform(
                InvertImageTransform(p_invert_image=1, p_synchronize_channels=0.5, p_per_channel=0.5),
                apply_probability=0.2,
            )
        )

        # ------------------------------------------------------------
        # 以下、デフォルトのデータ拡張にもある項目
        # ------------------------------------------------------------
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        transforms.append(RemoveLabelTansform(-1, 0))
        if is_cascaded:
            assert foreground_labels is not None, "We need foreground_labels for cascade augmentations"
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1, all_labels=foreground_labels, remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)), strel_size=(1, 8), p_per_label=1
                    ),
                    apply_probability=0.4,
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1,
                    ),
                    apply_probability=0.2,
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0,
                )
            )

        # ここでマスクをヒートマッヒートマップに変換している？
        # transforms.append(
        #     ConvertSegToRegrTarget(
        #         "EDT", gaussian_sigma=self.min_motor_distance // 3, edt_radius=self.min_motor_distance
        #     )
        # )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(deep_supervision_scales))

        transforms = ComposeTransforms(transforms)

        return transforms
