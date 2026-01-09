import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInferer
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import timm
# import timm_3d
from monai.networks.blocks import UnetResBlock
import numpy as np
import json


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


class CFG:
    # Data directories
    TEST_IMG_DIR = Path("./data/vesuvius-challenge-surface-detection/train_images")
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
    THRESHOLD = 0.1

    # Output settings
    OUTPUT_DIR = Path("./predictions")
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
            # ([3], 0),      # Flip H
            # ([4], 0),      # Flip W
            # ([2, 3], 0),   # Flip D+H
            # ([2, 4], 0),   # Flip D+W
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


def load_models_simple(model_paths, device):
    """Load models using Lightning's built-in checkpoint loading."""
    models = []
    for path in model_paths:
        print(f"Loading: {path}")
        # Lightning handles hyperparameters automatically
        # Force offline/backbone weights to avoid internet downloads
        model = SegmentationModule.load_from_checkpoint(
            path,
            map_location=device,
            pretrained_backbone=False,  # ensure timm does not download
        )
        model.to(device)
        model.eval()

        # Print model info
        model_type = getattr(model.hparams, 'model_type', 'unet')
        print(f"  Model type: {model_type}")
        if model_type == 'flex_unet':
            backbone = getattr(model.hparams, 'backbone', 'resnet18')
            print(f"  Backbone: {backbone}")

        models.append(model)
        print(f"  ✓ Loaded successfully")
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
def predict_volume_sliding_window(models, volume_tensor, device, use_tta=False):
    """
    Predict using ensemble of models with sliding window inference and optional TTA.

    Args:
        models: List of models
        volume_tensor: Input volume tensor (1, 1, D, H, W)
        device: Device to run inference on
        use_tta: Whether to use test-time augmentation

    Returns:
        Ensemble prediction probability map (D, H, W)
    """
    all_predictions = []

    # Create sliding window inferer
    inferer = SlidingWindowInferer(
        roi_size=CFG.ROI_SIZE,
        sw_batch_size=CFG.SW_BATCH_SIZE,
        overlap=CFG.OVERLAP,
        mode=CFG.SW_MODE,
        padding_mode=CFG.PADDING_MODE,
    )

    # Get TTA transforms
    tta_transforms = get_tta_transforms() if use_tta else [(None, 0)]

    # Get predictions from each model
    for i, model in enumerate(models):
        model.eval()

        # TTA loop
        tta_predictions = []
        for flip_dims, rotation_k in tta_transforms:
            # Apply TTA transform
            transformed_volume = apply_tta_transform(volume_tensor, flip_dims, rotation_k)

            # Move volume to device
            transformed_volume = transformed_volume.to(device)

            # Run sliding window inference
            with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                logits = inferer(transformed_volume, model)  # (1, 2, D, H, W) or list/tuple for deep supervision
            # deep supervision
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            # Apply softmax and get class 1 probability
            probs = torch.softmax(logits, dim=1)[:, 1]  # (1, D, H, W)
            probs = probs.squeeze(0).cpu().numpy()  # (D, H, W)

            # Reverse TTA transform
            probs_reversed = reverse_tta_transform(probs, flip_dims, rotation_k)
            tta_predictions.append(probs_reversed)

            # Free memory
            del transformed_volume, logits, probs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Average TTA predictions for this model
        model_pred = np.mean(tta_predictions, axis=0)
        all_predictions.append(model_pred)

    # Ensemble: average predictions from all models
    if len(all_predictions) > 1:
        ensemble_pred = np.mean(all_predictions, axis=0)
    else:
        ensemble_pred = all_predictions[0]

    return ensemble_pred



# ==============================================================================
# Model Loading - Simple approach using Lightning's load_from_checkpoint
# ==============================================================================

MODEL_PATHS = [
    './models/best-epoch=294-val_loss=0.3704-val_dice=0.5835.ckpt', #fold4
    './models/best-epoch=319-val_loss=0.3839-val_dice=0.5731.ckpt', #fold1
    './models/best-epoch=349-val_loss=0.3495-val_dice=0.6044.ckpt', #fold3
    './models/best-epoch=419-val_loss=0.3670-val_dice=0.5841.ckpt', #fold2
    './models/best-epoch449-val_loss0.3746-val_dice0.5755.ckpt' #fold0
]

def main():
    with open('./splits.json', "r") as f:
        val_splits = json.load(f)

    models = load_models_simple(MODEL_PATHS, CFG.DEVICE) if MODEL_PATHS else []
    selected_ids = [str(x) for x in val_splits[0]['val']]

    # Get test image files (.tif files)
    if CFG.TEST_IMG_DIR.exists():
        test_files = sorted([f for f in CFG.TEST_IMG_DIR.glob("*.tif")])
        test_files = [x for x in test_files if str(x).split('/')[-1].split('.tif')[0] in selected_ids]
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
    if len(test_files) > 0 and len(models) > 0:
        tif_dir = CFG.OUTPUT_DIR / "submission_tifs"
        processed_count = 0

        # Create dataset and dataloader
        test_dataset = InferenceDataset(test_files)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

        tta_status = "WITH TTA" if CFG.USE_TTA else "(no TTA)"
        print(f"\nRunning inference with sliding window {tta_status}...\n")

        if CFG.USE_TTA:
            tta_count = len(get_tta_transforms())
            print(f"Using {tta_count} TTA transforms per model")
            print(f"Total predictions per volume: {len(models)} models × {tta_count} TTA = {len(models) * tta_count}\n")

        for batch in tqdm(test_loader, desc="Processing volumes"):
            volume = batch['volume']  # (1, 1, D, H, W)
            vol_shape = tuple(batch['shape'][0].numpy())
            filename = batch['filename'][0]
            scroll_id = filename.replace('.tif', '')

            # Get ensemble prediction using sliding window (with optional TTA)
            ensemble_pred = predict_volume_sliding_window(models, volume, CFG.DEVICE, use_tta=CFG.USE_TTA)

            np.savez_compressed(f"{scroll_id}.tif", prob = ensemble_pred)

            # Free memory
            del ensemble_pred
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"\n✓ Inference complete! Saved {processed_count} .tif files to {tif_dir}")

