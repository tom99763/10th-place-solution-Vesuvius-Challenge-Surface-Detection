from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from ..procs.proc_data import *
from pathlib import Path

class ComposeDataset(Dataset):
    def __init__(self, cfg, id_list, train):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.proc_data = generate_transforms(self.cfg.data.transforms.train if train else self.cfg.data.transforms.val)
        self.data_path = Path(self.cfg.data_path)
        self.nnunet_path = Path(self.cfg.nnunet_path)
        self.compose_label_path = Path(self.cfg.compose_label_path)
        self.train = train

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        idx = self.id_list[idx]
        vol = load_volume(self.data_path / 'train_images' / f'{idx}.tif')
        mask = load_volume(self.data_path/'train_labels'/f'{idx}.tif')
        oof_mask = load_volume(Path(f'{self.cfg.oof_path1}/{idx}.tif'))
        raw = {"Image": vol, "Mask": mask, "Mask_OOF": oof_mask}
        data = self.proc_data(raw)
        if self.train:
            vol = torch.stack([_data['Image'] for _data in data], dim=0)
            mask = torch.stack([_data['Mask'] for _data in data], dim=0)
            mask_oof = torch.stack([_data['Mask_OOF'] for _data in data], dim=0)
        else:
            vol, mask, mask_oof = data['Image'], data['Mask'], data['Mask_OOF']
        return vol, mask, mask_oof


def collate_fn_train(batch):
    """
    batch: list of tuples (Image, Mask, Mask_OOF)
    """
    images = torch.cat([item[0] for item in batch], dim=0)
    masks = torch.cat([item[1] for item in batch], dim=0)
    mask_oof = torch.cat([item[2] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
        "Mask_OOF": mask_oof,
    }


def collate_fn_val(batch):
    """
    batch: list of tuples (Image, Mask, Mask_OOF)
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    mask_oof = torch.stack([item[2] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
        "Mask_OOF": mask_oof,
    }


class TomoDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_ids, val_ids):
        super().__init__()
        self.cfg = cfg
        self.train_ids = train_ids
        self.val_ids = val_ids

    def setup(self, stage: str = None):
        self.train_dataset = ComposeDataset(self.cfg, self.train_ids, True)
        self.val_dataset = ComposeDataset(self.cfg, self.val_ids, False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_train,
            persistent_workers = True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_val,
            persistent_workers = True
        )