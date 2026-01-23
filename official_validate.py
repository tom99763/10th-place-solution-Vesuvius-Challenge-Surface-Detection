# Standard library
import os
import sys
import json
import yaml
import shutil
import inspect
import warnings
import multiprocessing
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Union
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

# Third-party scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image, ImageSequence
from scipy.ndimage import convolve
from skimage.morphology import (
    skeletonize,
    binary_dilation,
    square,
)
from scipy.ndimage import convolve
import scipy.ndimage as ndimage
from skimage import morphology

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Training / utilities
from tqdm import tqdm
import pytorch_lightning as pl

# MONAI
from monai import transforms
from monai.networks.nets import UNet
from monai.networks.blocks import UnetResBlock
from monai.inferers import SlidingWindowInferer
from monai.inferers.inferer import SlidingWindowInfererAdapt

# nnU-Net / dynamic architectures
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

# Models / vision
import timm
from src.models.deformNet3d import *

# Configuration
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf

# Misc
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------
# configuration
# -------------
deformnet_ckpt_paths = [
    ('./models/deform-dynunet-v2-k3-s5-customFalse-fold0-epoch=19-val_bias_comp_metric=0.7012.ckpt', "cuda:0"),
    ('./models/deform-dynunet-v2-k3-s5-customFalse-fold1-epoch=34-val_bias_comp_metric=0.7526.ckpt', "cuda:1")
]
kernel_size0 = 3
kernel_size1 = 3
kernel_size2 = 3
sigma0 = 10
sigma1 = 5
sigma2 = 10
deformnet_threshold = 0.5
overlap = 0.5
num_iters = 1
min_obj_size = 200
closing_radius = 1
alpha_loop = 0.6
device = "cuda:0"


class CFG:
    # Data directories
    TEST_IMG_DIR = Path("data/vesuvius-challenge-surface-detection/train_images")
    MODEL_DIR = Path("")

    # Inference settings - sliding window
    ROI_SIZE = (160, 160, 160)  # Sliding window ROI size
    SW_BATCH_SIZE = 1  # Batch size for sliding window
    OVERLAP = 0.5  # Overlap ratio for sliding window
    SW_MODE = "gaussian"  # Mode: "constant", "gaussian"
    PADDING_MODE = "reflect"  # Padding mode

    # TTA settings
    USE_TTA = True  # Enable Test-Time Augmentation
    TTA_FLIPS = True  # Use flip augmentations
    TTA_ROTATIONS = True  # Use 90-degree rotations (warning: slower and memory intensive)

    # Model ensemble settings
    THRESHOLD = 0.3
    THRESHOLD_1ST_STAGE = 0.3
    THRESHOLD_2ND_STAGE = 0.5

    # Output settings
    OUTPUT_DIR = Path("./data/prob_predictions")
    SAVE_VISUALIZATIONS = True  # Set to True to save visualization images

    # Post-processing settings
    USE_POST_PROCESSING = False  # Enable mesh-based post-processing
    POST_PROCESS_MIN_CC_VOLUME = 3000  # Minimum connected component volume to keep

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Create output directories
CFG.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
(CFG.OUTPUT_DIR / "submission_tifs").mkdir(exist_ok=True, parents=True)

print(f"Device: {CFG.DEVICE}")
print(f"Threshold: {CFG.THRESHOLD}")
print(f"ROI Size: {CFG.ROI_SIZE}")
print(f"Overlap: {CFG.OVERLAP}")
print(f"Mode: {CFG.SW_MODE}")
print(f"TTA Enabled: {CFG.USE_TTA}")
if CFG.USE_TTA:
    print(f"  - Flips: {CFG.TTA_FLIPS}")
    print(f"  - Rotations: {CFG.TTA_ROTATIONS}")
print(f"Post-processing: {CFG.USE_POST_PROCESSING}")
if CFG.USE_POST_PROCESSING:
    print(f"  - Min CC Volume: {CFG.POST_PROCESS_MIN_CC_VOLUME}")
print(f"Visualizations: {CFG.SAVE_VISUALIZATIONS}")


# -------------
# Deformnet
# -------------

def load_model_from_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)


def load_average_model_soup(model, ckpt_paths):
    avg_state = None
    n = len(ckpt_paths)

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, weights_only=False)
        state_dict = ckpt["state_dict"]

        clean_state = OrderedDict()
        for k, v in state_dict.items():
            k = k.replace("model.", "") if k.startswith("model.") else k
            clean_state[k] = v.float()

        if avg_state is None:
            avg_state = OrderedDict({k: v.clone() for k, v in clean_state.items()})
        else:
            for k in avg_state:
                avg_state[k] += clean_state[k]

    # average
    for k in avg_state:
        avg_state[k] /= n

    model.load_state_dict(avg_state, strict=True)


def load_volume(path: Path) -> np.ndarray:
    """
    Load a multi-page TIFF into a 3D NumPy array: (slices, H, W)
    """
    try:
        with Image.open(path) as img:
            frames = [np.array(frame) for frame in ImageSequence.Iterator(img)]
        volume = np.stack(frames)
        return volume
    except Exception as e:
        raise RuntimeError(f"Error loading TIFF {path}: {e}")


def generate_transforms(
        transforms_config: list[dict],
) -> list[transforms.Transform]:
    transform_list = []
    # logger.debug(f"Generating {len(transforms_config)} transforms")

    for transform_config in transforms_config:
        transform_name = next(iter(transform_config))
        transform_kwargs = transform_config[transform_name]
        # logger.debug(
        #     f"Generating transform {transform_name} with kwargs {transform_kwargs}"
        # )
        transform: transforms.Transform = getattr(transforms, transform_name)(
            **transform_kwargs
        )  # type: ignore
        transform_list.append(transform)
    return transforms.Compose(transform_list)


def gaussian_kernel_3d(kernel_size=5, sigma=1.0, device="cuda"):
    """Returns a normalized 3D Gaussian kernel (1,1,K,K,K)."""
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_3d(x, kernel_size=7, sigma=10.0):
    """
    x: (B, C, D, H, W)
    """
    B, C, D, H, W = x.shape
    kernel = gaussian_kernel_3d(kernel_size, sigma, device=CFG.DEVICE)

    # shape: (C, 1, K, K, K)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size, kernel_size)

    # depthwise convolution
    return F.conv3d(x, kernel, padding=kernel_size // 2, groups=C)


def build_config(config_path, config_name, new_config_path, new_config_name):
    SRC = Path(config_path)
    DST = Path(new_config_path)
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC, DST, dirs_exist_ok=True)

    with initialize(version_base=None, config_path=new_config_name):
        cfg = compose(config_name=config_name)
    return cfg

def load_model_list(ckpts, cfg, model_arch):
    models = []
    for (ckpt_path, device) in tqdm(ckpts):
        m = model_arch(cfg).to(device)
        m.eval()
        load_model_from_checkpoint(m, ckpt_path)
        models.append(m)
    return models

@torch.no_grad()
def deformnet_step_worker(model, x, device, inferer):
    """
    Helper function to run a single model step on a specific GPU.
    """

    x_device = x.to(device)

    with torch.no_grad():
        pred_warped = inferer(x_device, model)

    return pred_warped.cpu()


@torch.no_grad()
def one_step_inference(prob_mask, vol, kernel_size, sigma,
                       deformnet_list, is_prob, apply_gaussian, threshold,
                       sliding_window_inferer, devices=["cuda:0", "cuda:1"]):
    # --- 1. Preprocessing (Centralized) ---
    # Perform the preparation on the device where data currently resides
    # (or force CPU if inputs might be on conflicting devices)

    if not is_prob:
        prob_mask = (prob_mask > threshold).float()

    if apply_gaussian:
        # Assuming gaussian_blur_3d is defined elsewhere
        prev_mask_pred = gaussian_blur_3d(prob_mask, kernel_size, sigma)
    else:
        prev_mask_pred = prob_mask

    x = torch.cat([vol, prev_mask_pred], dim=1)
    x = x.cpu()

    # --- 2. Parallel Inference ---
    future_predictions = []

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for i, model in enumerate(deformnet_list):
            # Round-robin assignment: Model 0 -> GPU 0, Model 1 -> GPU 1, etc.
            target_device = devices[i % len(devices)]

            future = executor.submit(
                deformnet_step_worker,
                model,
                x,
                target_device,
                sliding_window_inferer
            )
            future_predictions.append(future)

    # Gather results (blocks until all threads are done)
    predictions = [f.result() for f in future_predictions]

    # --- 3. Aggregation ---
    # predictions is now a list of CPU tensors
    return torch.cat(predictions, dim=0).mean(dim=0)


@torch.no_grad()
def inference_deformnet(data_path, prob_mask, deformnet_transforms, deformnet_list, sliding_window_inferer):
    print('start inference with deformnet')
    vol = load_volume(data_path)
    raw = {"Image": vol, "Mask_OOF": prob_mask}
    _data = deformnet_transforms(raw)
    vol, prob_mask = _data['Image'][None,], _data['Mask_OOF'][None,]
    vol = vol.to(device)
    prob_mask = prob_mask.to(device)
    prob_mask = prob_mask.float()
    # deformnetv2
    prediction_ensemble_v2 = one_step_inference(prob_mask, vol, kernel_size1,
                                                sigma1, deformnet_list, False, True, CFG.THRESHOLD,
                                                sliding_window_inferer
                                                )  # (c, d, h, w)
    prediction_ensemble = prediction_ensemble_v2
    # final
    segmentation_final = prediction_ensemble > deformnet_threshold
    segmentation_final = segmentation_final[0].cpu().numpy()
    return segmentation_final  # (D, H, W)




# -------------
# UNet + refine
# -------------
def get_conv(spatial_dims):
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d


def get_convtrans(spatial_dims):
    return nn.ConvTranspose3d if spatial_dims == 3 else nn.ConvTranspose2d


def upsample(x, size=None, spatial_dims=2):
    mode = "trilinear" if spatial_dims == 3 else "bilinear"
    return F.interpolate(x, size=size, mode=mode, align_corners=False)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, spatial_dims):
        super().__init__()
        ConvTrans = get_convtrans(spatial_dims)
        self.up = ConvTrans(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv_block = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=in_ch + skip_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            norm_name="batch"
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = upsample(x, size=skip.shape[2:], spatial_dims=3 if x.dim() == 5 else 2)
            x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_classes, spatial_dims):
        super().__init__()
        self.spatial_dims = spatial_dims
        enc = list(reversed(encoder_channels))
        self.blocks = nn.ModuleList()
        in_ch = enc[0]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = enc[i + 1] if (i + 1) < len(enc) else 0
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch, spatial_dims))
            in_ch = out_ch
        Conv = get_conv(spatial_dims)
        self.final = Conv(in_ch, n_classes, kernel_size=1)

    def forward(self, features, original_size=None):
        feats = list(reversed(features))
        x = feats[0]
        for i, block in enumerate(self.blocks):
            skip = feats[i + 1] if (i + 1) < len(feats) else None
            x = block(x, skip)
        x = self.final(x)
        if original_size is not None and x.shape[2:] != original_size:
            x = upsample(x, size=original_size, spatial_dims=self.spatial_dims)
        return x

def create_residual_unet(
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 320, 320),
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        deep_supervision=False,
):
    """Create ResidualEncoderUNet matching nnUNet 3d_fullres configuration.

    Args:
        in_channels: Input channels (default: 1)
        out_channels: Output channels (default: 2)
        channels: Feature channels at each level (tuple of ints)
        strides: Strides for downsampling at each level (tuple of ints)
        n_blocks_per_stage: Number of residual blocks per stage (tuple of ints)
        deep_supervision: Whether to use deep supervision

    Returns:
        ResidualEncoderUNet model
    """
    # Number of stages in decoder is len(channels) - 1
    n_conv_per_stage_decoder = [1] * (len(channels) - 1)

    model = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=len(channels),
        features_per_stage=channels,
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=out_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    return model


# ==============================================================================
# SegmentationModule - matches training code structure
# ==============================================================================
class SegmentationModule(pl.LightningModule):
    """
    Lightning module matching training code structure.
    Supports both 'unet' and 'flex_unet' model types.
    Uses self.model (not self.model1) to match checkpoint keys.
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 1),
            dropout=0.1,
            model_type="unet",  # 'unet' or 'flex_unet'
            # FlexibleUNet parameters
            backbone="resnet18",
            pretrained_backbone=False,
            drop_path_rate=0.0,
            **kwargs  # Accept extra hparams from checkpoint
    ):
        super().__init__()
        self.save_hyperparameters()

        # Match training code: uses self.model
        if model_type == "unet":
            self.model = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                num_res_units=2,
                dropout=dropout,
                norm='instance',
                act='relu',
            )
        elif model_type == "flex_unet":
            self.model = FlexibleUNet(
                backbone=backbone,
                in_channels=in_channels,
                n_classes=out_channels,
                is_3d=True,
                pretrained=pretrained_backbone,
                drop_path_rate=drop_path_rate,
            )
        elif model_type == "res_unet":
            if create_residual_unet is None:
                raise ImportError("res_unet requires src.models.residual_unet and dynamic_network_architectures")

            # Extract specific parameters for ResidualUNet
            n_blocks = kwargs.get('n_blocks_per_stage', (1, 3, 4, 6, 6, 6))
            use_ds = kwargs.get('use_deep_supervision', False)

            self.model = create_residual_unet(
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                n_blocks_per_stage=n_blocks,
                deep_supervision=use_ds
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}. Use 'unet', 'flex_unet' or 'res_unet'")

    def forward(self, x):
        return self.model(x)


# ==============================================================================
# SegmentationModule - matches training code structure
# ==============================================================================
class SegmentationModule2ndStage(pl.LightningModule):
    """
    Lightning module matching training code structure.
    Uses self.model to match checkpoint keys.
    """

    def __init__(
            self,
            in_channels=2,
            out_channels=2,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(1, 2, 2, 2, 2, 2),
            **kwargs  # Accept extra hparams from checkpoint
    ):
        super().__init__()
        self.save_hyperparameters()

        n_blocks = kwargs.get('n_blocks_per_stage', (1, 3, 4, 6, 6, 6))
        use_ds = kwargs.get('use_deep_supervision', False)

        self.model = create_residual_unet(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            n_blocks_per_stage=n_blocks,
            deep_supervision=use_ds
        )

    def forward(self, x):
        return self.model(x)



def load_array(path, fmt):
    """Load array from various formats"""
    if fmt == "tiff" or fmt == "tif":
        return tiff.imread(path)
    elif fmt == "npy":
        return np.load(path)
    elif fmt == "npz":
        return np.load(path)["arr_0"]
    elif fmt == "rle":
        rle = np.load(path)
        shape = tuple(rle["shape"])
        vals = rle["vals"]
        runs = rle["runs"]
        flat = np.repeat(vals, runs)
        return flat.reshape(shape)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def normalize_volume(volume):
    volume = volume.astype(np.float32)

    volume = volume / 255.0

    return volume


def rle_encode(mask):
    """Run-length encoding for submission"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def apply_tta_transform(volume, flip_dims=None, rotation_k=0):
    """
    Apply TTA transformation to volume.

    Args:
        volume: Input volume tensor (1, 1, D, H, W)
        flip_dims: List of dimensions to flip (e.g., [2, 3, 4] for D, H, W)
        rotation_k: Number of 90-degree rotations (0-3) on H-W plane

    Returns:
        Transformed volume
    """
    transformed = volume.clone()

    # Apply flips
    if flip_dims:
        for dim in flip_dims:
            transformed = torch.flip(transformed, dims=[dim])

    # Apply rotation on H-W plane (dims 3 and 4)
    if rotation_k > 0:
        transformed = torch.rot90(transformed, k=rotation_k, dims=[3, 4])

    return transformed


def reverse_tta_transform(prediction, flip_dims=None, rotation_k=0):
    """
    Reverse TTA transformation on prediction.

    Args:
        prediction: Prediction tensor (D, H, W) or (1, D, H, W)
        flip_dims: List of dimensions to flip (adjusted for prediction dims)
        rotation_k: Number of 90-degree rotations to reverse

    Returns:
        Reversed prediction
    """
    # Handle both (D, H, W) and (1, D, H, W) shapes
    if prediction.ndim == 3:
        pred = torch.from_numpy(prediction).unsqueeze(0)  # (1, D, H, W)
        squeeze_output = True
    else:
        pred = torch.from_numpy(prediction)
        squeeze_output = False

    # Reverse rotation first (opposite direction)
    if rotation_k > 0:
        pred = torch.rot90(pred, k=(4 - rotation_k), dims=[2, 3])

    # Reverse flips
    if flip_dims:
        # Adjust flip dims for (1, D, H, W) shape
        adjusted_dims = [d - 1 for d in flip_dims] if flip_dims else None
        if adjusted_dims:
            for dim in adjusted_dims:
                pred = torch.flip(pred, dims=[dim])

    result = pred.squeeze(0).numpy() if squeeze_output else pred.numpy()
    return result


def get_tta_transforms():
    """
    Get list of TTA transformations to apply.

    Returns:
        List of (flip_dims, rotation_k) tuples
    """
    transforms = [
        (None, 0),  # Original (no transform)
    ]

    if CFG.TTA_FLIPS:
        transforms.extend([
            ([2], 0),  # Flip D
            ([3], 0),  # Flip H
            ([4], 0),  # Flip W
            ([2, 3], 0),  # Flip D+H
            ([2, 4], 0),  # Flip D+W
            # ([3, 4], 0),   # Flip H+W
            # ([2, 3, 4], 0) # Flip all
        ])

    if CFG.TTA_ROTATIONS:
        # Add rotation augmentations (90, 180, 270 degrees)
        transforms.extend([
            (None, 1),  # 90° rotation
            # (None, 2),  # 180° rotation
            # (None, 3),  # 270° rotation
        ])

    return transforms


def load_models_simple(model_paths):
    """Load models using Lightning's built-in checkpoint loading, handling EMA weights."""
    models = []
    for (path, device) in model_paths:
        print(f"Loading: {path}")

        # First, load the checkpoint to check for EMA weights
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Check if checkpoint has EMA weights
        has_ema = any(k.startswith('ema.module.') for k in state_dict.keys())
        use_ema_hparam = checkpoint.get('hyper_parameters', {}).get('use_ema', False)

        # Create model from checkpoint
        model = SegmentationModule.load_from_checkpoint(
            path,
            map_location=device,
            pretrained_backbone=False,  # ensure no downloads
            strict=False
        )

        # If EMA weights exist, extract and load them into model
        if has_ema:
            print(f"  Found EMA weights in checkpoint, extracting...")
            ema_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('ema.module.'):
                    # Remove 'ema.module.' prefix to get 'model.*' key
                    new_key = 'model.' + k[len('ema.module.'):]
                    ema_state_dict[new_key] = v

            # Load EMA weights into the model
            missing, unexpected = model.load_state_dict(ema_state_dict, strict=False)
            if missing:
                # Filter out non-model keys from missing (like loss functions, etc.)
                model_missing = [k for k in missing if k.startswith('model.')]
                if model_missing:
                    print(f"  Warning: Missing EMA keys: {model_missing[:5]}...")
            print(f"  ✓ Loaded EMA weights into model")

        model.to(device)
        model.eval()

        # Print model info
        model_type = getattr(model.hparams, 'model_type', 'unet')
        print(f"  Model type: {model_type}")
        if model_type == 'flex_unet':
            backbone = getattr(model.hparams, 'backbone', 'resnet18')
            print(f"  Backbone: {backbone}")
        if has_ema or use_ema_hparam:
            print(f"  ✓ Using EMA weights for inference")
        else:
            print(f"  Standard weights (no EMA)")
        print(f"  ✓ Loaded successfully")

        models.append(model)
    print(f"\n✓ Total models loaded: {len(models)}")
    return models


def load_models_2nd_stage(model_paths, device):
    """Load models using Lightning's built-in checkpoint loading, handling EMA weights."""
    models = []
    for path in model_paths:
        print(f"Loading: {path}")

        # First, load the checkpoint to check for EMA weights
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Check if checkpoint has EMA weights
        has_ema = any(k.startswith('ema.module.') for k in state_dict.keys())
        use_ema_hparam = checkpoint.get('hyper_parameters', {}).get('use_ema', False)

        # Create model from checkpoint
        model = SegmentationModule2ndStage.load_from_checkpoint(
            path,
            map_location=device,
            strict=False
        )

        # If EMA weights exist, extract and load them into model
        if has_ema:
            print(f"  Found EMA weights in checkpoint, extracting...")
            ema_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('ema.module.'):
                    # Remove 'ema.module.' prefix to get 'model.*' key
                    new_key = 'model.' + k[len('ema.module.'):]
                    ema_state_dict[new_key] = v

            # Load EMA weights into the model
            missing, unexpected = model.load_state_dict(ema_state_dict, strict=False)
            if missing:
                # Filter out non-model keys from missing (like loss functions, etc.)
                model_missing = [k for k in missing if k.startswith('model.')]
                if model_missing:
                    print(f"  Warning: Missing EMA keys: {model_missing[:5]}...")
            print(f"  ✓ Loaded EMA weights into model")

        model.to(device)
        model.eval()

        # Print model info
        model_type = getattr(model.hparams, 'model_type', 'unet')
        print(f"  Model type: {model_type}")
        if model_type == 'flex_unet':
            backbone = getattr(model.hparams, 'backbone', 'resnet18')
            print(f"  Backbone: {backbone}")
        if has_ema or use_ema_hparam:
            print(f"  ✓ Using EMA weights for inference")
        else:
            print(f"  Standard weights (no EMA)")
        print(f"  ✓ Loaded successfully")

        models.append(model)
    print(f"\n✓ Total models loaded: {len(models)}")
    return models



class InferenceDataset(Dataset):
    """Dataset for inference on test volumes - no resizing"""

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load original volume
        vol = load_array(img_path, "tif")
        vol_shape = vol.shape  # Get actual input shape

        # Normalize volume (no resizing)
        vol_normalized = normalize_volume(vol)

        # Convert to tensor (add channel dimension)
        vol_tensor = torch.from_numpy(vol_normalized).float().unsqueeze(0)  # (1, D, H, W)

        return {
            'volume': vol_tensor,
            'shape': vol_shape,
            'filename': img_path.name
        }


def custom_collate_fn(batch):
    """Custom collate function"""
    item = batch[0]
    return {
        'volume': item['volume'].unsqueeze(0),  # Add batch dimension
        'shape': torch.tensor([item['shape']]),
        'filename': [item['filename']]
    }


@torch.no_grad()
def _predict_single_model_worker(model, volume_tensor, target_device, inferer, tta_transforms):
    """
    Worker function to run inference for a single model on a specific GPU.
    """
    # 1. Move model to the assigned GPU
    # model.to(target_device)
    model.eval()

    tta_predictions = []

    # 2. Run TTA Loop (Same logic as original)
    for flip_dims, rotation_k in tta_transforms:
        transformed_volume = apply_tta_transform(volume_tensor, flip_dims, rotation_k)

        # Move data to the SAME device as the model
        transformed_volume = transformed_volume.to(target_device)

        with torch.amp.autocast(device_type='cuda'):
            logits = inferer(transformed_volume, model)

        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        probs = torch.softmax(logits, dim=1)[:, 1]
        probs = probs.squeeze(0).cpu().numpy()  # Move result back to CPU immediately

        probs_reversed = reverse_tta_transform(probs, flip_dims, rotation_k)
        tta_predictions.append(probs_reversed)

        del transformed_volume, logits, probs
        # Optional: Empty cache only if strictly necessary to avoid synchronization overhead
        # torch.cuda.empty_cache()

    # Average TTA predictions for this specific model
    return np.mean(tta_predictions, axis=0)


@torch.no_grad()
def predict_volume_sliding_window(models, volume_tensor, devices, use_tta=False):
    """
    Args:
        models: List of models (e.g., [model_A, model_B])
        devices: List of devices to use (e.g., ['cuda:0', 'cuda:1'])
    """
    # Create shared inferer (stateless)
    inferer = SlidingWindowInferer(
        roi_size=CFG.ROI_SIZE,
        sw_batch_size=CFG.SW_BATCH_SIZE,
        overlap=CFG.OVERLAP,
        mode=CFG.SW_MODE,
        padding_mode=CFG.PADDING_MODE,
    )

    tta_transforms = get_tta_transforms() if use_tta else [(None, 0)]

    # Ensure volume is on CPU initially to avoid GPU-to-GPU copies
    volume_tensor = volume_tensor.cpu()

    future_results = []

    # 3. Execute in Parallel
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for i, model in enumerate(models):
            # Assign model to a device in round-robin fashion
            assigned_device = devices[i % len(devices)]

            # Submit task to thread pool
            future = executor.submit(
                _predict_single_model_worker,
                model,
                volume_tensor,
                assigned_device,
                inferer,
                tta_transforms
            )
            future_results.append(future)

    # 4. Gather Results
    # .result() blocks until the specific thread is finished
    all_predictions = [f.result() for f in future_results]

    # Ensemble averaging
    if len(all_predictions) > 1:
        ensemble_pred = np.mean(all_predictions, axis=0)
    else:
        ensemble_pred = all_predictions[0]

    return ensemble_pred



# ==============================================================================
# Model Loading - Simple approach using Lightning's load_from_checkpoint
# ==============================================================================

MODEL_PATHS = [
    ("./models/best-epoch449-val_loss0.3746-val_dice0.5755.ckpt", "cuda"),
    ("./models/best-epoch369-val_loss0.3769-val_dice0.5727.ckpt", "cuda")
]


MODEL_PATHS_2ND_STAGE = [
    "./models/2nd_stage_best-epoch104-val_dice0.5820-val_loss0.3709.ckpt"
]


def postprocess_mask_voxel(mask, min_cc_volume=3000, verbose=True):
    """
    Post-process binary mask using voxel-based operations:
    1. Remove small connected components
    2. Fill small holes/gaps using binary closing

    Returns:
        processed_mask, stats_dict (if verbose) or just processed_mask
    """
    if mask.sum() == 0:
        if verbose:
            return mask, {"original_cc": 0, "removed_cc": 0, "kept_cc": 0}
        return mask

    try:
        mask_bool = mask.astype(bool)

        # Count original connected components
        labeled_original, num_original = ndimage.label(mask_bool)

        # Get sizes of each component
        component_sizes = ndimage.sum(mask_bool, labeled_original, range(1, num_original + 1))

        # Count how many are below threshold
        num_small = np.sum(component_sizes < min_cc_volume)
        num_kept = num_original - num_small

        # Remove small connected components
        cleaned = morphology.remove_small_objects(mask_bool, min_size=min_cc_volume)

        # Fill holes / close gaps
        # closed = ndimage.binary_closing(cleaned, structure=np.ones((3,3,3)))

        if verbose:
            stats = {
                "original_cc": num_original,
                "removed_cc": int(num_small),
                "kept_cc": int(num_kept),
                "component_sizes": component_sizes
            }
            return cleaned.astype(np.uint8), stats

        return cleaned.astype(np.uint8)

    except Exception as e:
        print(f"    Error in voxel post-processing: {e}")
        if verbose:
            return mask, {"error": str(e)}
        return mask



def main():
    with open('./splits.json', "r") as f:
        val_splits = json.load(f)

    models_1st_stage = load_models_simple(MODEL_PATHS)
    models_2nd_stage = load_models_2nd_stage(MODEL_PATHS_2ND_STAGE, CFG.DEVICE)
    cfg = build_config("./configs", 'config_deform_v2',
                       "./configs_1", 'configs_1'
                       )
    deformnet_list = load_model_list(deformnet_ckpt_paths, cfg, DeformDynUnetV2)
    deformnet_transforms = generate_transforms(cfg.data.transforms.test)
    sliding_window_inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.input_size, sw_batch_size=1, overlap=overlap, mode="gaussian",
        progress=True
    )

    for i in range(3,5):
        print(f'fold {i}.......')
        selected_ids = [str(x) for x in val_splits[i]['val']]

        # Get test image files (.tif files)
        if CFG.TEST_IMG_DIR.exists():
            test_files = sorted([f for f in CFG.TEST_IMG_DIR.glob("*.tif")])
            test_files = [x for x in test_files if str(x).split("/")[-1].split('.tif')[0] in selected_ids]
            print(f"Found {len(test_files)} test .tif files")

            if len(test_files) > 0:
                print("\nTest files:")
                for f in test_files[:5]:
                    print(f"  - {f.name}")
                if len(test_files) > 5:
                    print(f"  ... and {len(test_files) - 5} more")
        else:
            print(f"Warning: Test directory not found at {CFG.TEST_IMG_DIR}")
            print("Please update CFG.TEST_IMG_DIR to point to the correct directory")
            test_files = []

        # Run inference and save predictions directly to .tif files
        if len(test_files) > 0:
            # Create dataset and dataloader
            test_dataset = InferenceDataset(test_files)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

            for batch in tqdm(test_loader, desc="Processing volumes"):
                volume = batch['volume']  # (1, 1, D, H, W)
                vol_shape = tuple(batch['shape'][0].numpy())
                filename = batch['filename'][0]
                scroll_id = filename.replace('.tif', '')

                # ------------------
                # 1st stage inference
                # ------------------
                first_stage_probs = predict_volume_sliding_window(
                    models_1st_stage,
                    volume,
                    devices=["cuda"],
                    use_tta=CFG.USE_TTA
                )

                # # Binarize 1st stage output for 2nd stage input
                first_stage_binary = (first_stage_probs > CFG.THRESHOLD_1ST_STAGE).astype(np.float32)

                # ------------------
                # 2nd stage inference
                # ------------------
                first_stage_binary_tensor = torch.from_numpy(first_stage_binary).unsqueeze(0).unsqueeze(0)
                to_second_stage_pred = first_stage_binary_tensor
                volume_2ch = torch.cat([volume, to_second_stage_pred], dim=1)
                second_stage_probs = predict_volume_sliding_window(
                    models_2nd_stage,
                    volume_2ch,
                    devices=["cuda"],
                    use_tta=False
                )
                ensemble_pred = inference_deformnet(CFG.TEST_IMG_DIR / filename, second_stage_probs,
                                                    deformnet_transforms, deformnet_list, sliding_window_inferer)
                binary_mask = ensemble_pred.astype(np.uint8)

                if CFG.USE_POST_PROCESSING:
                    try:
                        tqdm.write(f"  Post-processing {scroll_id}...", end="")
                        binary_mask, stats = postprocess_mask_voxel(
                            binary_mask,
                            min_cc_volume=CFG.POST_PROCESS_MIN_CC_VOLUME,
                            verbose=True
                        )
                        tqdm.write(f" ✓ (removed {stats['removed_cc']}/{stats['original_cc']} components)")
                    except Exception as e:
                        tqdm.write(f"  ⚠ Warning: Post-processing failed for {scroll_id}: {e}")
                        tqdm.write("  Saving original mask without post-processing...")


                # Free memory
                del second_stage_probs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    main()

