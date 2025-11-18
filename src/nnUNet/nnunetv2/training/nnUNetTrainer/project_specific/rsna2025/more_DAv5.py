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
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform, OneOfTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainer_onlyMirror01,
)
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    _brightnessadditive_localgamma_transform_scale,
    _brightness_gradient_additive_max_strength,
    _local_gamma_gamma,
)


class MirrorLRSwapLabelsTransform(BasicTransform):
    """
    Left-right mirror along a specified spatial axis and swap label IDs for left/right classes to keep semantics.

    This assumes tensors with shapes (b, c, x, y, z) for 3D or (b, c, x, y) for 2D. The left-right axis is given
    w.r.t. spatial axes (0-based), so for (x, y, z) use axis=2 if z corresponds to left-right in your data pipeline.
    The transform always applies the flip and label remap; control probability via RandomTransform wrapper.
    """

    def __init__(
        self,
        lr_axis: int = 2,
        label_pairs_to_swap: List[Tuple[int, int]] = None,
        apply_to_keys: Tuple[str, ...] = ("image", "segmentation"),
    ) -> None:
        super().__init__()
        self.lr_axis = lr_axis
        # Default pairs for RSNA 2025 cerebral arteries: (Right, Left)
        if label_pairs_to_swap is None:
            label_pairs_to_swap = [
                (3, 4),  # Posterior Communicating Artery R/L
                (5, 6),  # Infraclinoid ICA R/L
                (7, 8),  # Supraclinoid ICA R/L
                (9, 10),  # MCA R/L
                (11, 12),  # ACA R/L
            ]
        self.label_pairs_to_swap = label_pairs_to_swap
        self.apply_to_keys = apply_to_keys

    def __call__(self, **data_dict):
        # determine dim and target flip dim index
        if "image" not in data_dict:
            return data_dict

        tensor = data_dict["image"]
        # Expect 4D (c,x,y,z)
        if tensor.ndim != 4:
            return data_dict

        spatial_dims = tensor.ndim - 1
        # Only operate if the requested axis exists (typically 3D with axis=2)
        if not (0 <= self.lr_axis < spatial_dims):
            return data_dict

        flip_dim = 1 + self.lr_axis  # account for (c, ...)

        for k in self.apply_to_keys:
            if k not in data_dict:
                continue
            t = data_dict[k]
            if not torch.is_tensor(t):
                # convert numpy to torch tensor to use flip, then back
                is_numpy = True
                t_torch = torch.from_numpy(t)
            else:
                is_numpy = False
                t_torch = t

            # flip
            t_torch = torch.flip(t_torch, dims=(flip_dim,))

            # label swap only for segmentation (assumes integer labels)
            if k == "segmentation":
                # seg may be (1, x, y, z); operate on the label map regardless of channel dim
                # We do swapping on a copy of the flipped tensor to avoid in-place clashes
                s = t_torch.clone()
                out = t_torch.clone()
                for a, b in self.label_pairs_to_swap:
                    out[s == a] = b
                    out[s == b] = a
                t_torch = out

            if is_numpy:
                data_dict[k] = t_torch.numpy()
            else:
                data_dict[k] = t_torch

        return data_dict


class RSNA2025Trainer_moreDAv5(nnUNetTrainer_onlyMirror01):
    """
    v5: Inherits v3-style heavy DA and adds explicit left-right mirroring with label swapping.
    """

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

        # どれかの軸だけ大きくダウンサンプリング
        transforms.append(
            OneOfTransform(
                [
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[0, 1],
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[0, 2],
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[1, 2],
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                ]
            )
        )
        # どれかの軸だけ大きくダウンサンプリング
        transforms.append(
            OneOfTransform(
                [
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[0, 1],
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[0, 2],
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[1, 2],
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                ]
            )
        )

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

        # add explicit left-right mirroring with label swapping (axis=2 assumed LR)
        transforms.append(
            RandomTransform(
                MirrorLRSwapLabelsTransform(lr_axis=2),
                apply_probability=0.5,
            )
        )

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
