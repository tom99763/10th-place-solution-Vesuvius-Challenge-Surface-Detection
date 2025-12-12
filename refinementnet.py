import torch
import torch.nn as nn
from monai.networks.nets import SegResNetDS
from monai.utils.misc import ensure_tuple_rep

import pytorch_lightning as pl
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch

import tifffile
from torch.utils.data import Dataset
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from monai.data import list_data_collate

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    ConcatItemsd, RandCropByPosNegLabeld, RandRotate90d, ToTensord, EnsureTyped
)

torch.set_float32_matmul_precision('medium')

def get_transforms(mode="train"):
    """
    Args:
        mode: 'train' for augmentation + cropping, 'val' for full volume normalization only.
    """
    transforms_list = [
        # 1. Add Channel Dimension: (Depth, H, W) -> (1, Depth, H, W)
        EnsureChannelFirstd(keys=["image", "noisy_mask", "label"], channel_dim="no_channel"),
        
        # 2. Normalize Intensity (Image only, masks stay 0/1)
        ScaleIntensityd(keys=["image"]),
        
        # 3. Concatenate Image + Noisy Mask -> (2, Depth, H, W)
        ConcatItemsd(keys=["image", "noisy_mask"], name="model_input", dim=0),
        
        # 4. Ensure data types (float32 for images, often uint8/int for labels)
        EnsureTyped(keys=["model_input", "label"]),
    ]
    
    # Training-specific Augmentations
    if mode == "train":
        transforms_list.extend([
            # Crop 96x96x96 cubes centered around the object
            RandCropByPosNegLabeld(
                keys=["model_input", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1, neg=1, num_samples=1, 
                image_key="model_input",
                image_threshold=0,
            ),
            # Random Rotations
            RandRotate90d(keys=["model_input", "label"], prob=0.5, spatial_axes=(0, 2)),
        ])
    
    transforms_list.append(ToTensord(keys=["model_input", "label"]))
    
    return Compose(transforms_list)



class IgnoreRegionDiceCELoss(nn.Module):
    def __init__(self, ignore_index=2, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # 1. Cross Entropy Part
        # PyTorch's CrossEntropyLoss has a built-in 'ignore_index' argument
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # 2. Dice Part
        # We use MONAI's DiceLoss but we will feed it masked inputs manually
        self.dice_loss = DiceLoss(
            include_background=False, # Focus on FG
            to_onehot_y=False,        # We will handle one-hot manually to mask it
            softmax=True,             # Convert logits to probs
            reduction="mean",
            smooth_nr=1e-5,           # Smoothing to prevent div-by-zero
            smooth_dr=1e-5
        )

    def forward(self, logits, labels):
        """
        logits: (B, C, D, H, W) - Raw model output
        labels: (B, 1, D, H, W) - Ground truth with 0, 1, and 2 (ignore)
        """
        
        # --- Part A: Cross Entropy Loss ---
        # CE expects labels as (B, D, H, W) long tensor (no channel dim)
        labels_sq = labels.squeeze(1).long()
        ce = self.ce_loss(logits, labels_sq)
        
        # --- Part B: Masked Dice Loss ---
        
        # 1. Create the Validity Mask (1 where we care, 0 where label is 2)
        valid_mask = (labels != self.ignore_index).float()
        
        # 2. Clean the Labels for One-Hot Encoding
        # We temporarily set label 2 to 0 to avoid errors during one-hot encoding.
        # It doesn't matter what we set it to, because we will multiply by valid_mask anyway.
        labels_clean = labels.clone()
        labels_clean[labels == self.ignore_index] = 0
        
        # 3. One-Hot Encode Labels
        # monai.networks.utils.one_hot is useful, or just standard torch
        # Assuming 2 classes (BG, FG)
        num_classes = logits.shape[1]
        labels_onehot = torch.nn.functional.one_hot(labels_clean.squeeze(1).long(), num_classes=num_classes)
        labels_onehot = labels_onehot.permute(0, 4, 1, 2, 3).float() # (B, C, D, H, W)
        
        # 4. Get Probabilities from Logits
        probs = torch.softmax(logits, dim=1)
        
        # 5. Apply the Mask to BOTH Probabilities and One-Hot Labels
        # This effectively removes the 'ignore' pixels from the Dice Sums (Intersection & Union)
        # We expand mask to match channel dims
        valid_mask_expanded = valid_mask.expand_as(probs)
        
        probs_masked = probs * valid_mask_expanded
        labels_masked = labels_onehot * valid_mask_expanded
        
        # 6. Calculate Dice
        # We pass the ALREADY masked tensors. 
        # Note: We must disable softmax=True here because we did it manually above
        # Note: We must disable to_onehot_y=True here because we did it manually
        dice = self.dice_loss(probs_masked, labels_masked)
        
        # --- Combine ---
        total_loss = (self.ce_weight * ce) + (self.dice_weight * dice)
        
        return total_loss

class RefinementSegResNet(pl.LightningModule):
    def __init__(self, lr=1e-4, input_size=(96, 96, 96)):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Initialize SegResNetVAE
        # Note: input_image_size is CRITICAL here.
        self.model = SegResNetDS(
            spatial_dims=3,
            in_channels=2,   # Image + Noisy Mask
            out_channels=2,  # Background, Foreground
        )
        
        # 2. Loss & Metrics
        self.loss_function = IgnoreRegionDiceCELoss(
            ignore_index=2, 
            ce_weight=1.0, 
            dice_weight=1.0
        )
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

    def forward(self, x):
        # SegResNetVAE returns (seg_output, vae_loss) in training
        # But for inference, we usually only care about the seg_output.
        # However, calling self.model(x) ALWAYS calculates VAE loss if use_vae=True.
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["model_input"], batch["label"]
        
        # Forward pass returns TWO things
        seg_output = self(inputs)
        
        # 1. Calculate Segmentation Loss (Dice + CE)
        seg_loss = self.loss_function(seg_output, labels)
        
        # 2. Total Loss = Seg Loss + (Weight * VAE Loss)
        # vae_loss is already a scalar tensor calculated inside the model
        total_loss = seg_loss
        
        # Logging
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["model_input"], batch["label"]
        

        outputs = sliding_window_inference(
            inputs, 
            (96, 96, 96), 
            4, 
            self, # Use the wrapper to discard VAE loss
            overlap=0
        )
        
        loss = self.loss_function(outputs, labels)
        
        # Metrics
        preds = [torch.argmax(i, dim=0, keepdim=True) for i in decollate_batch(outputs)]
        targets = [i for i in decollate_batch(labels)]
        self.dice_metric(y_pred=preds, y=targets)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_mean_dice", metric, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-5)

class TiffRefinementDataset(Dataset):
    def __init__(self, foldid, transform=None, mode="train"):
        """
        Args:
            img_dir (str): Path to folder containing original .tif volumes
            noisy_mask_dir (str): Path to folder containing noisy prediction .tif
            label_dir (str): Path to folder containing ground truth .tif
            transform (callable): MONAI transforms
        """
        self.transform = transform

        with open("./nnunet/preprocessed/Dataset900_VesuviusScroll/splits_final.json", "r") as f:
            self.split_plan = json.load(f)

        self.ids = self.split_plan[foldid]["train"] if mode == "train" else self.split_plan[foldid]["val"]
        
        self.foldid = foldid

        self.imgdir = Path("./data/train_images")
        self.lbldir = Path("./data/train_labels")
        self.prddir = Path("./data/predmasks")
        self.mode = mode
       
    

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        id = self.ids[idx]

        # 1. Load the 3D files
       
        img_path = self.imgdir / f"{id}.tif"
        label_path = self.lbldir / f"{id}.tif"
        noisy_path = self.prddir / f"{id}.tif"

        
        img_vol = tifffile.imread(img_path).astype(np.float32).copy()
        noisy_vol = tifffile.imread(noisy_path).astype(np.float32).copy()
        label_vol = tifffile.imread(label_path).astype(np.float32).copy()
        
        # 2. Create the dictionary for MONAI
        data_dict = {
            "image": img_vol,
            "noisy_mask": noisy_vol,
            "label": label_vol
        }
        
        # 3. Apply Transforms
        if self.transform:
            data_dict = self.transform(data_dict)

        if self.mode == "train":
            for x in data_dict:
                del x["image"], x["noisy_mask"]


        # 4. Return
        # If training, 'model_input' is (2, 96, 96, 96)
        # If validation, 'model_input' is (2, D, H, W) full volume
        return data_dict


def get_dataloaders(
    foldid,
    batch_size=2, 
    num_workers=4
):

    # --- Training Setup ---
    train_ds = TiffRefinementDataset(
        foldid,
        transform=get_transforms(mode="train"),
        mode = "train"
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=list_data_collate, # Handles stacking different sized crops if needed
        pin_memory=True
    )

    val_ds = TiffRefinementDataset(
        foldid,
        transform=get_transforms(mode="val"),
        mode="val"
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=list_data_collate
    )

    return train_loader, val_loader

def main():
    # 1. Setup Paths
    log_dir = "./logs"
    foldid = 0
    
    # 2. Init DataLoaders
    train_loader, val_loader = get_dataloaders(
        foldid, 
        batch_size=8, 
        num_workers=8
    )
    
    # 3. Init Model
    model = RefinementSegResNet(lr=1e-4)
    
    # 4. Setup Loggers & Callbacks
    
    # TensorBoard Logger
    # Logs will be saved at: ./logs/refinement_project/version_X
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="refinement_project",
        default_hp_metric=False
    )
    
    # Checkpointing: Save best model based on Dice Score
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_dice",
        mode="max",
        filename="best-checkpoint-{epoch:02d}-{val_mean_dice:.4f}",
        save_top_k=1,
        save_last=True
    )
    
    # 5. Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback],
        
        # Logging settings
        log_every_n_steps=10,    # Log to Tensorboard every 10 steps
        enable_progress_bar=True # Enables the TQDM bar
    )
    
    # 6. Start Training
    print("Starting Training...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
