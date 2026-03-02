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
from concurrent.futures import ThreadPoolExecutor
from dynamic_network_architectures.architectures.primus import PrimusV2B

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def create_primus_v2(
        in_channels=1,
        out_channels=2,
        input_shape = 160,
        patch_embed_size = 8,
        drop_path_rate = 0.0,
):
    model = PrimusV2B(in_channels,
                      out_channels,
                      (patch_embed_size, patch_embed_size, patch_embed_size),
                      (input_shape, input_shape, input_shape),
                      drop_path_rate=drop_path_rate)
    return model


# ==============================================================================
# SegmentationModule - matches training code structure
# ==============================================================================
class SegmentationModule(pl.LightningModule):
    """
    Lightning module matching training code structure.
    Uses self.model to match checkpoint keys.
    """

    def __init__(
            self,
            in_channels=4,
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


class SegmentationModulePrimusV2(pl.LightningModule):
    """
    Lightning module matching training code structure.
    Uses self.model to match checkpoint keys.
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(1, 2, 2, 2, 2, 2),
            **kwargs  # Accept extra hparams from checkpoint
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_primus_v2(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        return self.model(x)



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
    REPEAT = 2

    # TTA settings
    USE_TTA = True # Enable Test-Time Augmentation
    TTA_FLIPS = True  # Use flip augmentations
    TTA_ROTATIONS = True  # Use 90-degree rotations (warning: slower and memory intensive)

    # Model ensemble settings
    THRESHOLD = 0.3
    THRESHOLD_1ST_STAGE = 0.3
    THRESHOLD_2ND_STAGE = 0.5

    # Output settings
    OUTPUT_DIR = Path(f"./data/2st_stage_oof_repeat{REPEAT}")
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
        model = SegmentationModulePrimusV2.load_from_checkpoint(
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


def load_models_2nd_stage(model_paths):
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

        model = SegmentationModule.load_from_checkpoint(
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
    ("./models/1st-stage/primus-v2/best-epoch=379-val_loss=0.3646-val_dice=0.5936.ckpt", "cuda"),
    ("./models/1st-stage/primus-v2/best-epoch=404-val_loss=0.3655-val_dice=0.5881.ckpt", "cuda"),
]


MODEL_PATHS_2ND_STAGE = [
    ("./models/2st-stage/refine-v2-4ch/fold_0/best-epoch=134-val_dice=0.6015-val_loss=0.3528.ckpt", "cuda"),
    ("./models/2st-stage/refine-v2-4ch/fold_1/best-epoch=119-val_dice=0.6102-val_loss=0.3507.ckpt", "cuda"),
    ("./models/2st-stage/refine-v2-4ch/fold_2/fold_2best-epoch139-val_dice0.6118-val_loss0.3393.ckpt", "cuda"),
    ("./models/2st-stage/refine-v2-4ch/fold_3/best-epoch=104-val_dice=0.6296-val_loss=0.3275.ckpt", "cuda"),
    ("./models/2st-stage/refine-v2-4ch/fold_4/best-epoch=139-val_dice=0.6252-val_loss=0.3342.ckpt", "cuda"),
]

def main():
    with open('./splits.json', "r") as f:
        val_splits = json.load(f)

    if not os.path.exists('./data/refine-v2-4ch-oof'):
        os.makedirs('./data/refine-v2-4ch-oof')

    primus_v2 = load_models_simple(MODEL_PATHS)

    for i in range(3):
        models_2nd_stage = load_models_2nd_stage([MODEL_PATHS_2ND_STAGE[i]])
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
                first_stage_probs_primus_v2 = predict_volume_sliding_window(
                    primus_v2,
                    volume,
                    devices=["cuda"],
                    use_tta=False
                )

                # # Binarize 1st stage output for 2nd stage input
                first_stage_binary_resenc = np.load(f'./models/1st-stage/resenc/1st_stage_oof/{scroll_id}.npy')
                first_stage_binary_resenc = (first_stage_binary_resenc > CFG.THRESHOLD_1ST_STAGE).astype(np.float32)
                first_stage_binary_primus = np.load(f'./models/1st-stage/primus/Primus_1st_stage_cache/{scroll_id}_probs.npy')
                first_stage_binary_primus = (first_stage_binary_primus > CFG.THRESHOLD_1ST_STAGE).astype(np.float32)
                first_stage_binary_primus_v2 = (first_stage_probs_primus_v2 > CFG.THRESHOLD_1ST_STAGE).astype(np.float32)

                # ------------------
                # 2nd stage inference
                # ------------------
                first_stage_binary_resenc = torch.from_numpy(first_stage_binary_resenc).unsqueeze(0).unsqueeze(0)
                first_stage_binary_primus = torch.from_numpy(first_stage_binary_primus).unsqueeze(0).unsqueeze(0)
                first_stage_binary_primus_v2 = torch.from_numpy(first_stage_binary_primus_v2).unsqueeze(0).unsqueeze(0)

                input_2nd_stage_4ch = torch.cat([volume,
                                                 first_stage_binary_resenc,
                                                 first_stage_binary_primus,
                                                 first_stage_binary_primus_v2
                                                 ], dim=1)
                p_ref_next_4ch = predict_volume_sliding_window(
                    models_2nd_stage ,
                    input_2nd_stage_4ch,
                    devices=["cuda"],
                    use_tta=False
                )

                np.save(f'./data/refine-v2-4ch-oof/{scroll_id}.npy', p_ref_next_4ch.astype('float16'))

                # Free memory
                del p_ref_next_4ch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    main()
