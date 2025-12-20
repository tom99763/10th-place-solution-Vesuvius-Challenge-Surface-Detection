from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from ..procs.proc_data import generate_transforms, load_volume
import numpy as np

class ComposeDataset(Dataset):
    def __init__(self, cfg, id_list, train: bool):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.train = train
        self.proc_data = generate_transforms(cfg.data.transforms.train if train else cfg.data.transforms.val)
        self.data_path = Path(cfg.data_path)
        self.compose_label_path = Path(cfg.compose_label_path)
        self.oof_path = Path(cfg.oof_path1)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        idx = self.id_list[idx]
        vol = load_volume(self.data_path / 'train_images' / f'{idx}.tif')
        mask = load_volume(self.data_path / 'train_labels' / f'{idx}.tif')
        oof_mask = load_volume(self.oof_path / f'{idx}.tif')

        if self.train:
            c = np.load(self.compose_label_path / f'{idx}.npz', mmap_mode ='r')
            c1 = c['thickness']
            c2 = c['sdf']
            c3 = c['normals']

            raw = {
                "Image": vol,
                "Mask": mask,
                "Mask_OOF": oof_mask,
                "C1": c1,
                "C2": c2,
                "C3": c3
            }

            data = self.proc_data(raw)
            vol = torch.stack([d['Image'] for d in data], dim=0)
            mask = torch.stack([d['Mask'] for d in data], dim=0)
            mask_oof = torch.stack([d['Mask_OOF'] for d in data], dim=0)
            c1 = torch.stack([d['C1'] for d in data], dim=0)
            c2 = torch.stack([d['C2'] for d in data], dim=0)
            c3 = torch.stack([d['C3'] for d in data], dim=0)

            return vol, mask, mask_oof, c1, c2, c3

        else:
            raw = {
                "Image": vol,
                "Mask": mask,
                "Mask_OOF": oof_mask
            }
            data = self.proc_data(raw)
            vol = data['Image']
            mask = data['Mask']
            mask_oof = data['Mask_OOF']
            return vol, mask, mask_oof


def collate_fn_train(batch):
    images = torch.cat([item[0] for item in batch], dim=0)
    masks = torch.cat([item[1] for item in batch], dim=0)
    mask_oof = torch.cat([item[2] for item in batch], dim=0)
    c1 = torch.cat([item[3] for item in batch], dim=0)
    c2 = torch.cat([item[4] for item in batch], dim=0)
    c3 = torch.cat([item[5] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
        "Mask_OOF": mask_oof,
        "C1": c1,
        "C2": c2,
        "C3": c3
    }


def collate_fn_val(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    mask_oof = torch.stack([item[2] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
        "Mask_OOF": mask_oof
    }


class TomoDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_ids, val_ids):
        super().__init__()
        self.cfg = cfg
        self.train_ids = train_ids
        self.val_ids = ['1407735'] #val_ids

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