from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from ..procs.proc_data import generate_transforms, load_volume, load_sparse_tensor
import numpy as np
import pytorch_lightning as pl

class ComposeDataset(Dataset):
    def __init__(self, cfg, id_list, train: bool):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.train = train
        self.proc_data = generate_transforms(cfg.data.transforms.train if train else cfg.data.transforms.val)
        self.data_path = Path(cfg.data_path)
        self.cdl_label_path = Path(self.cfg.cdl_label_path)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        idx = self.id_list[idx]
        vol = load_volume(self.data_path / 'train_images' / f'{idx}.tif')
        mask = load_volume(self.data_path / 'train_labels' / f'{idx}.tif')
        Z = load_sparse_tensor(str(self.cdl_label_path/f'{idx}.npz'))
        if self.train:
            raw = {
                "Image": vol,
                "Mask": mask,
                "Z": Z
            }
            data = self.proc_data(raw)
            vol = torch.stack([d['Image'] for d in data], dim=0)
            mask = torch.stack([d['Mask'] for d in data], dim=0)
            Z = torch.stack([d['Z'] for d in data], dim=0)
            return vol, mask, Z

        else:
            raw = {
                "Image": vol,
                "Mask": mask,
            }
            data = self.proc_data(raw)
            vol = data['Image']
            mask = data['Mask']
            return vol, mask


def collate_fn_train(batch):
    images = torch.cat([item[0] for item in batch], dim=0)
    masks = torch.cat([item[1] for item in batch], dim=0)
    Z = torch.cat([item[2] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
        "Z": Z,
    }


def collate_fn_val(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
    }


class TomoDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_ids, val_ids):
        super().__init__()
        self.cfg = cfg
        self.train_ids = train_ids
        self.val_ids = val_ids[:5]

    def setup(self, stage: str = None):
        self.train_dataset = ComposeDataset(self.cfg, self.train_ids, train=True)
        self.val_dataset = ComposeDataset(self.cfg, self.val_ids, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_train,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_val,
            persistent_workers=True
        )