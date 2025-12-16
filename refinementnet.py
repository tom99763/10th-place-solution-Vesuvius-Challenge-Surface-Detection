import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from monai import transforms as MT


import tifffile
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from typing import Tuple, Dict
from monai.losses import DiceCELoss, TverskyLoss

from monai.networks.nets import SwinUNETR

import re
from pathlib import Path
from typing import List, Union, Tuple, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException


import json

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")


class SurfaceDataset3D(Dataset):
    """3D Surface Detection Dataset.

    Updated to support volume-based loading.
    Optimized for faster Torch conversion.
    Supports .tif, .npy, .npz formats.
    """

    def __init__(self, ids, imgdir, lbldir):
        super().__init__()
        self.ids = ids
        self.imgdir = imgdir
        self.lbldir = lbldir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        id = self.ids[idx]
        # Load raw data -> (D, H, W)
        image, mask = self._load_from_raw(id)

        # Optimization: Convert directly to Tensor to avoid intermediate numpy float64 copies
        # 1. Convert raw uint8/uint16 -> Tensor
        # 2. Cast to float16
        # 3. Scale
        # 4. Add channel dim
        image_t = torch.from_numpy(image).half().div_(255.0).unsqueeze(0)
        mask_t = torch.from_numpy(mask).long().unsqueeze(0)

        return image_t, mask_t, id

    def _load_file(self, path: Path) -> np.ndarray:
        """Helper to load generic file formats."""
        return tifffile.imread(str(path))

    def _load_from_raw(
        self,
        id: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        image_path = self.imgdir / f"{id}.tif"
        image_volume = self._load_file(image_path)

        label_path = self.lbldir / f"{id}.tif"
        label_volume = self._load_file(label_path)

        return image_volume, label_volume


def custom_collate(batch):
    """Custom collate to handle variable size 3D volumes.
    Returns a list of items instead of stacking them, allowing GPU resizing later.
    """
    return batch


class SurfaceDataModule(pl.LightningDataModule):
    """Lightning DataModule for Surface Detection.

    Handles all data loading, splitting, and dataloader creation.
    Updated to use MONAI 3D augmentations on GPU with dynamic resizing.
    """

    def __init__(
        self,
        foldid,
        imgdir: Path,
        lbldir: Path,
        volume_shape: Tuple[int, int, int],
        batch_size: int = 1,
        num_workers: int = 2,
    ):
        super().__init__()
        self.foldid = foldid
        self.imgdir = imgdir
        self.lbldir = lbldir

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.volume_shape = volume_shape
        # Define GPU-based augmentations using MONAI
        # 1. Resize to target shape (trilinear for image, nearest for label)
        # 2. Apply Augmentations
        self.gpu_augments = MT.Compose(
            [
                MT.Resized(
                    keys=["image", "label"],
                    spatial_size=self.volume_shape,
                    mode=["trilinear", "nearest"],
                ),
                MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                MT.RandRotated(
                    keys=["image", "label"],
                    range_x=0.1,
                    range_y=0.1,
                    range_z=0.1,
                    prob=0.3,
                    keep_size=True,
                    mode=["bilinear", "nearest"],
                ),
                MT.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                MT.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
            ]
        )
        # Validation transforms: Just Resize (for image AND label)
        self.val_augments = MT.Compose(
            [
                MT.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    spatial_size=self.volume_shape,
                    label_key="label",
                    image_key="image",
                    pos=1, neg=1, num_samples=1, 
                    image_threshold=0,
                )
            ]
        )
        # Validation transforms for Image ONLY (for test set where labels are None)
        self.val_image_augments = MT.Compose(
            [
                MT.Resized(
                    keys=["image"], spatial_size=self.volume_shape, mode=["trilinear"]
                )
            ]
        )

        with open(
            "./nnunet/preprocessed/Dataset900_VesuviusScroll/splits_final.json", "r"
        ) as f:
            self.split_plan = json.load(f)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        print(f"\nSetting up training data...")

        foldids = self.split_plan[self.foldid]

        trainids = foldids["train"]
        valids = foldids["val"]

        print(f"Train ids: {len(trainids)}")
        print(f"Val ids: {len(valids)}")

        # Create train dataset
        self.train_dataset = SurfaceDataset3D(
            trainids,
            imgdir=self.imgdir,
            lbldir=self.lbldir,
        )
        # Create validation dataset
        self.val_dataset = SurfaceDataset3D(
            trainids,
            imgdir=self.imgdir,
            lbldir=self.lbldir,
        )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader with custom collate for variable sizes."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.num_workers > 0),
            collate_fn=custom_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader with custom collate for variable sizes."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.num_workers > 0),
            collate_fn=custom_collate,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply MONAI GPU-accelerated 3D augmentations to the batch."""
        # If custom_collate is used, batch is a list of tuples [(x, y, id), ...]
        if not isinstance(batch, list):
            return super().on_after_batch_transfer(batch, dataloader_idx)

        x_list, y_list, frag_ids = [], [], []
        # Determine device to ensure we process on GPU
        # self.trainer.strategy.root_device is reliable in Lightning
        device = (
            self.trainer.strategy.root_device
            if self.trainer
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Select transform
        transforms = self.gpu_augments if self.trainer.training else self.val_augments

        for item in batch:
            x, y, frag_id = item

            # # IMPORTANT: Explicitly move to GPU now.
            # # This ensures the Resized and other transforms run on VRAM, avoiding CPU RAM spikes.
            # x = x.to(device, non_blocking=True)
            # y = y.to(device, non_blocking=True)

            data = {"image": x, "label": y}
            # Apply transforms (Resize + Augments)
            data = transforms(data)

            x_list.append(data["image"])
            y_list.append(data["label"])
            frag_ids.append(frag_id)

        # Stack into tensors -> (B, C, D, H, W)
        return torch.stack(x_list), torch.stack(y_list), frag_ids


class SurfaceSegmentation3D(pl.LightningModule):
    """3D Surface Segmentation using a custom network.

    Key Design Choices:
    - **Loss**: Combined DiceCELoss + TverskyLoss to handle structural imbalance.
    - **Metrics**: Manual computation of Dice and IoU ignoring class 2.
    """

    def __init__(
        self,
        net: nn.Module,
        out_channels: int = 2,
        spatial_dims: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        ignore_index_val: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net_module = net
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ignore_index_val = ignore_index_val

        # Loss function configuration
        # TverskyLoss with alpha=0.7 emphasizes minimizing False Negatives (Recall)
        self.criterion_tversky = TverskyLoss(
            softmax=True,
            to_onehot_y=False,
            include_background=True,
            alpha=0.7,
            beta=0.3,
        )
        # DiceCELoss combines Dice Loss and Cross Entropy Loss
        self.criterion_dice_ce = DiceCELoss(
            softmax=True,
            to_onehot_y=False,
            include_background=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_module(x)

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss excluding class 2 (unlabeled). Optimized for GPU."""
        # targets shape: (B, 1, D, H, W)
        mask = targets != self.ignore_index_val
        # Prepare targets for One-Hot Encoding (replace ignore index with 0 temporary)
        targets_sq = targets.squeeze(1)
        targets_clean = torch.where(
            mask.squeeze(1), targets_sq, torch.tensor(0, device=targets.device)
        )
        # One-Hot Encode
        targets_onehot = torch.nn.functional.one_hot(
            targets_clean.long(), num_classes=self.hparams.out_channels
        ).float()
        if self.hparams.spatial_dims == 3:
            targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3)
        else:
            targets_onehot = targets_onehot.permute(0, 3, 1, 2)
        # Mask One-Hot Targets
        targets_masked_ohe = targets_onehot * mask.half()

        # Compute both losses and sum them
        loss_tversky = self.criterion_tversky(logits, targets_masked_ohe)
        loss_dice_ce = self.criterion_dice_ce(logits, targets_masked_ohe)

        return loss_tversky + loss_dice_ce

    def _compute_metrics(
        self, preds_logits: torch.Tensor, targets_class_indices: torch.Tensor
    ) -> dict:
        preds_proba = torch.softmax(preds_logits, dim=1)
        preds_hard = torch.argmax(preds_proba, dim=1, keepdim=True)
        valid_mask = (
            targets_class_indices != self.ignore_index_val
        ).float()  # (B, 1, D, H, W)
        num_classes = preds_logits.shape[1]  # This will be 2 (background, foreground)
        dice_scores_per_class = []
        iou_scores_per_class = []

        for i in range(num_classes):
            pred_class_i = (preds_hard == i).float()  # (B, 1, D, H, W)
            target_class_i = (targets_class_indices == i).float()  # (B, 1, D, H, W)

            pred_class_i_valid = pred_class_i * valid_mask
            target_class_i_valid = target_class_i * valid_mask

            intersection = (pred_class_i_valid * target_class_i_valid).sum()
            union_sum_dice = pred_class_i_valid.sum() + target_class_i_valid.sum()
            union_sum_iou = (
                pred_class_i_valid.sum() + target_class_i_valid.sum() - intersection
            )
            dice = (2 * intersection + 1e-8) / (union_sum_dice + 1e-8)
            iou = (intersection + 1e-8) / (union_sum_iou + 1e-8)
            dice_scores_per_class.append(dice)
            iou_scores_per_class.append(iou)

        mean_dice = torch.mean(torch.stack(dice_scores_per_class))
        mean_iou = torch.mean(torch.stack(iou_scores_per_class))
        return {"dice": mean_dice, "iou": mean_iou}

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets, _ = batch
        logits = self(inputs)
        loss = self._compute_loss(logits, targets)

        metrics = self._compute_metrics(logits, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_dice", metrics["dice"], on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_iou", metrics["iou"], on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets, _ = batch
        logits = self(inputs)
        loss = self._compute_loss(logits, targets)

        metrics = self._compute_metrics(logits, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_dice", metrics["dice"], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val_iou", metrics["iou"], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # Cosine Annealing Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else MAX_EPOCHS,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def predict_step(self, batch: Tuple, batch_idx: int) -> Dict:
        inputs, _, frag_id = batch
        logits = self(inputs)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        return {"prediction": pred_class, "fragment_id": frag_id}


def get_best_checkpoint(
    checkpoint_dirs: Union[str, Path, List[Union[str, Path]]],
    name: str = "",
) -> Tuple[str, float]:
    """Finds the checkpoint with the highest val_dice score across multiple directories."""
    # Normalize input to a list of Paths
    if not isinstance(checkpoint_dirs, list):
        checkpoint_dirs = [checkpoint_dirs]
    checkpoint_dirs = [d for d in checkpoint_dirs if Path(d).exists()]
    if not checkpoint_dirs:
        print("No valid folder provided.")
        return None, None

    checkpoints = []
    # Regex for val_dice
    pattern = re.compile(r"val_dice=?([0-9]+\.[0-9]+)")
    # Iterate over all files in all valid directories
    for path in [f for d in checkpoint_dirs for f in Path(d).glob(f"{name}*.ckpt")]:
        match = pattern.search(path.name)
        if not match:
            continue
        checkpoints.append((float(match.group(1)), str(path)))

    if not checkpoints:
        print("No valid checkpoints found.")
        return None, None

    # Sort by score descending so the best is first
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = checkpoints[0]
    print(f"Found {len(checkpoints)} checkpoints.")
    print(f"Best  (Score={best_score}): {Path(best_path)}")
    return best_path, best_score


def main():
    # Create DataModule
    datamodule = SurfaceDataModule(
        foldid=FOLDID,
        imgdir=TRAIN_IMAGES_DIR,
        lbldir=TRAIN_LABELS_DIR,
        volume_shape=MODEL_INPUT_SIZE,
    )
    datamodule.setup()

    # Initialize SwinUNETR model
    net = SwinUNETR(
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_v2=True,
        drop_rate=0.2,
        attn_drop_rate=0.2,
        dropout_path_rate=0.2,
    )
    net_name = net.__class__.__name__
    model = SurfaceSegmentation3D(net=net)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename=net_name + "-{epoch:02d}-{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=3,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_dice", patience=10, mode="max", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # TensorBoard Logger
    # Logs will be saved at: ./logs/refinement_project/version_X
    logger = TensorBoardLogger(
        save_dir="./logs", name="refinement_project", default_hp_metric=False
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        precision="16-mixed",
        log_every_n_steps=1,
        enable_progress_bar=True,
        accumulate_grad_batches=18,
        gradient_clip_val=1.0,  # Clips gradient norm to 1.0 to prevent exploding gradients
    )

    ckpt_path, ckpt_score = get_best_checkpoint(
        [OUTPUT_DIR, CHECKPOINT_DIR], name=net_name
    )

    # Train
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    except MisconfigurationException as ex:
        print(ex)


if __name__ == "__main__":

    FOLDID = 0

    # Paths
    DATA_DIR = Path("./data")
    CHECKPOINT_DIR = "./model-zoo"

    TRAIN_IMAGES_DIR = Path("./data/train_images")
    TRAIN_LABELS_DIR = Path("./data/train_labels")
    TEST_IMAGES_DIR = DATA_DIR / "test_images"
    OUTPUT_DIR = Path(".")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_INPUT_SIZE = (
        128,
        128,
        128,
    )  # (depth, height, width) - resize volumes to this
    IN_CHANNELS = 1  # grayscale
    OUT_CHANNELS = 2  # background + papyrus (ignore class 2)

    # Training
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    MAX_EPOCHS = 50
    LEARNING_RATE = 2e-3

    main()
