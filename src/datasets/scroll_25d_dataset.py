from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path

from ..procs.proc_data import generate_transforms


class ScrollDataset25D(Dataset):
    """
    2.5D dataset:
    Loads precomputed NPY files shaped (3, H, W) for Image and (1,H,W) or (H,W) for Mask.
    """
    def __init__(self, cfg, id_list):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list

        # MONAI transform pipeline
        self.proc_vol_and_mask = generate_transforms(self.cfg.transforms)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        sample_id = self.id_list[idx]
        data_path = Path(self.cfg.data_path)

        # -----------------------------
        # Load 2.5D NPY image
        # Shape: (3, H, W)
        # -----------------------------
        img_path = data_path / f"train_images/{sample_id}.npy"
        vol = np.load(img_path).astype(np.float32)

        # -----------------------------
        # Load mask depending on stage
        # -----------------------------
        if self.cfg.stage == "1":
            mask_path = data_path / f"rough_masks/rough_mask_{sample_id}.npy"
        elif self.cfg.stage == "2":
            mask_path = data_path / f"train_labels/{sample_id}.npy"
        else:
            raise Exception("Invalid cfg.stage (must be '1' or '2')")

        mask = np.load(mask_path).astype(np.float32)

        # Ensure (1, H, W) for MONAI
        if mask.ndim == 2:
            mask = mask[None]

        transformed = self.proc_vol_and_mask({"Image": vol, "Mask": mask})
        return transformed["Image"], transformed["Mask"]


class ScrollDataModule25D(pl.LightningDataModule):
    """
    Lightning DataModule wrapper for ScrollDataset25D.
    """
    def __init__(self, cfg, id_list, train_idx, val_idx):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.train_idx = train_idx
        self.val_idx = val_idx

    def setup(self, stage=None):
        self.train_dataset = ScrollDataset25D(
            self.cfg.data, self.id_list[self.train_idx]
        )
        self.val_dataset = ScrollDataset25D(
            self.cfg.data, self.id_list[self.val_idx]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
