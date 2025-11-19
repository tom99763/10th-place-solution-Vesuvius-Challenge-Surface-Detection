from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.procs.proc_data import *



class ScrollDataset25D(Dataset):
    """
    2.5D dataset for nested structure:
    
    data/
      train_images_25d_1/<video_id>/<slice.npy>
      train_labels_25d_1/<video_id>/<slice.npy>

    id_list contains items like:
        "video_001/slice_010.npy"
        "video_001/slice_011.npy"
        "video_002/slice_020.npy"
    """

    def __init__(self, cfg, id_list, mode="train"):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.data_path = Path(self.cfg.data_path)
        self.mode=mode
        if mode == 'train':
            self.proc_vol_and_mask = generate_transforms(cfg.data.transforms.train)
        else:
            self.proc_vol_and_mask = generate_transforms(cfg.data.transforms.val)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        slice_path = self.id_list[idx]
        # example: "video_001/slice_010.npy"

        # -----------------------------
        # Load NPY (3,H,W) and (H,W) or (1,H,W)
        # -----------------------------
        vol = np.load(slice_path, mmap_mode='r').astype(np.float32)
        mask_path = Path(str(slice_path).replace("train_images_25d_1", "train_labels_25d_1"))
        mask_path = Path(str(mask_path).replace("img", "mask"))
        mask = np.load(mask_path, mmap_mode='r').astype(np.float32)

        # Ensure mask has channel dimension (1,H,W)
        if mask.ndim == 2:
            mask = mask[None]
        elif mask.shape[0] != 1:
            # allow mask to also be (3,H,W)
            
            pass

        transformed = self.proc_vol_and_mask({"Image": vol, "Mask": mask})

        return transformed["Image"], transformed["Mask"]



class ScrollDataModule25D(pl.LightningDataModule):
    def __init__(self, cfg, train_ids, val_ids):
        super().__init__()
        self.cfg = cfg
        self.train_ids = train_ids
        self.val_ids = val_ids

    def setup(self, stage=None):
        self.train_dataset = ScrollDataset25D(
            self.cfg, self.train_ids ,mode="train"
        )
        self.val_dataset = ScrollDataset25D(
            self.cfg, self.val_ids, mode="val" )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )