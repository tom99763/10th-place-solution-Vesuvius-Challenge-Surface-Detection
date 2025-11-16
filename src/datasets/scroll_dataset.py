from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from ..procs.proc_data import *
from pathlib import Path

class TomoDataset(Dataset):
    def __init__(self, cfg, id_list):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.proc_vol_and_mask = generate_transforms(self.cfg.transforms)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        data_path = Path(self.cfg.data_path)

        vol = load_volume(data_path/f'train_images/{id}.tif')

        if self.cfg.stage == '1':
            mask = load_volume(data_path/f'rough_masks/rough_mask_{id}.tif')
        elif self.cfg.stage == '2':
            mask = load_volume(data_path / f'train_labels/{id}.tif')
        else:
            raise Exception('Invalid stage')

        transformed = self.proc_vol_and_mask({'Image': vol, 'Mask': mask})
        return transformed['Image'], transformed['Mask']


class TomoDataModule(pl.LightningDataModule):
    def __init__(self, cfg, id_list, train_idx, val_idx):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.train_idx = train_idx
        self.val_idx = val_idx

    def setup(self, stage: str = None):
        self.train_dataset = TomoDataset(self.cfg,
                                         self.id_list[self.train_idx],
                                         )
        self.val_dataset = TomoDataset(self.cfg,
                                       self.id_list[self.val_idx],
                                       )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )