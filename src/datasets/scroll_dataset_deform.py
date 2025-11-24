from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from ..procs.proc_data import *
from pathlib import Path

class DeformDataset(Dataset):
    def __init__(self, cfg, id_list):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.proc_data = generate_transforms(self.cfg.transforms)
        self.data_path = Path(self.cfg.data_path)
        self.nnunet_path = Path(self.cfg.nnunet_path)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        idx = self.id_list[idx]
        vol = load_volume(self.data_path / 'train_images' / f'{idx}.tif')
        mask = load_volume(self.data_path/'train_labels'/f'{idx}.tif')
        pred_mask = np.load(self.nnunet_path/f'{idx}.npz')
        raw = {"Image": vol, "Mask": mask, "Mask_OOF": pred_mask}
        data = self.proc_data(raw)
        return data["Image"], data["Mask"], data["Mask_OOF"]


def collate_fn(batch):
    """
    batch: list of tuples (Image, Mask, Mask_OOF)
    """
    images = torch.concat([item[0] for item in batch], dim=0) #(batch * num_pos_sample, c, d, h, w)
    masks = torch.concat([item[1] for item in batch], dim=0)
    mask_oof = torch.concat([item[2] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
        "Mask_OOF": mask_oof
    }


class TomoDataModule(pl.LightningDataModule):
    def __init__(self, cfg, id_list, train_idx, val_idx):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.train_idx = train_idx
        self.val_idx = val_idx

    def setup(self, stage: str = None):
        self.train_dataset = DeformDataset(self.cfg,
                                         self.id_list[self.train_idx],
                                         )
        self.val_dataset = DeformDataset(self.cfg,
                                       self.id_list[self.val_idx],
                                       )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )