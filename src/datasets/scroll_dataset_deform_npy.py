from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from ..procs.proc_data import generate_transforms
from pathlib import Path


class DeformDatasetNpy(Dataset):
    """DeformNet dataset that loads .npy files for images, labels, and OOF masks."""

    def __init__(self, cfg, id_list, train):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.proc_data = generate_transforms(
            self.cfg.data.transforms.train if train else self.cfg.data.transforms.val
        )
        self.image_dir = Path(self.cfg.image_dir)
        self.label_dir = Path(self.cfg.label_dir)
        self.oof_path = Path(self.cfg.oof_path)
        self.train = train

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        sample_id = self.id_list[idx]

        vol = np.load(self.image_dir / f'{sample_id}.npy')
        mask = np.load(self.label_dir / f'{sample_id}.npy')
        pred_mask = np.load(self.oof_path / f'{sample_id}_probs.npy').astype(np.float32)

        if getattr(self.cfg, 'binarize_oof', False):
            pred_mask = (pred_mask > self.cfg.threshold).astype(np.float32)

        raw = {"Image": vol, "Mask": mask, "Mask_OOF": pred_mask}
        data = self.proc_data(raw)

        if self.train:
            vol = torch.stack([_data['Image'] for _data in data], dim=0)
            mask = torch.stack([_data['Mask'] for _data in data], dim=0)
            mask_oof = torch.stack([_data['Mask_OOF'] for _data in data], dim=0)
        else:
            vol, mask, mask_oof = data['Image'], data['Mask'], data['Mask_OOF']

        return vol, mask, mask_oof


def collate_fn_train(batch):
    images = torch.cat([item[0] for item in batch], dim=0)
    masks = torch.cat([item[1] for item in batch], dim=0)
    mask_oof = torch.cat([item[2] for item in batch], dim=0)
    return {"Image": images, "Mask": masks, "Mask_OOF": mask_oof}


def collate_fn_val(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    mask_oof = torch.stack([item[2] for item in batch], dim=0)
    return {"Image": images, "Mask": masks, "Mask_OOF": mask_oof}


class TomoDataModuleNpy(pl.LightningDataModule):
    def __init__(self, cfg, train_ids, val_ids):
        super().__init__()
        self.cfg = cfg
        self.train_ids = train_ids
        self.val_ids = val_ids

    def setup(self, stage=None):
        self.train_dataset = DeformDatasetNpy(self.cfg, self.train_ids, True)
        self.val_dataset = DeformDatasetNpy(self.cfg, self.val_ids, False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_train,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_val,
            persistent_workers=True,
        )
