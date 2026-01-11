from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from ..procs.proc_data import *
from pathlib import Path
import random


class DeformDataset(Dataset):
    def __init__(self, cfg, id_list, train):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.proc_data = generate_transforms(self.cfg.data.transforms.train if train else self.cfg.data.transforms.val)
        self.data_path = Path(self.cfg.data_path)
        self.nnunet_path = Path(self.cfg.nnunet_path)
        self.train = train

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        idx = self.id_list[idx]
        vol = np.load(f'../../vesuvius_challenge/vesuvius_challenge/train_images_npy/{idx}.npy')
        mask = np.load(f'../../vesuvius_challenge/vesuvius_challenge/train_labels_npy/{idx}.npy')
        skel = np.load(f'../../vesuvius_challenge/vesuvius_challenge/train_skeletons_npy/{idx}.npy')
        if self.cfg.is_prob_oof_mask:
            pred_mask = np.load(f'{self.cfg.oof_path}/{idx}.npz', mmap_mode='r')
            pred_mask = pred_mask['prob'] #(d, h, w)
        else:
            pred_mask = load_volume(Path(f'{self.cfg.oof_path}/{idx}.tif'))
            #pred_mask = np.load(Path(f'{self.cfg.oof_path}/{idx}.npy'))

        #pred_mask = pad_skeleton_3d(pred_mask)

        raw = {"Image": vol, "Mask": mask, "Mask_OOF": pred_mask, "Skel": skel}
        data = self.proc_data(raw)
        if self.train:
            vol = torch.stack([_data['Image'] for _data in data], dim=0)
            mask = torch.stack([_data['Mask'] for _data in data], dim=0)
            mask_oof = torch.stack([_data['Mask_OOF'] for _data in data], dim=0)
            skel = torch.stack([_data['Skel'] for _data in data], dim=0)
        else:
            vol, mask, mask_oof, skel = data['Image'], data['Mask'], data['Mask_OOF'], data['Skel']
        return vol, mask, mask_oof, skel


def collate_fn_train(batch):
    """
    batch: list of tuples (Image, Mask, Mask_OOF)
    """
    images = torch.cat([item[0] for item in batch], dim=0)
    masks = torch.cat([item[1] for item in batch], dim=0)
    mask_oof = torch.cat([item[2] for item in batch], dim=0)
    skel = torch.cat([item[3] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
        "Mask_OOF": mask_oof,
        "Skel": skel
    }


def collate_fn_val(batch):
    """
    batch: list of tuples (Image, Mask, Mask_OOF)
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    mask_oof = torch.stack([item[2] for item in batch], dim=0)
    skel = torch.stack([item[3] for item in batch], dim=0)

    return {
        "Image": images,
        "Mask": masks,
        "Mask_OOF": mask_oof,
        "Skel": skel
    }


class TomoDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_ids, val_ids):
        super().__init__()
        self.cfg = cfg
        for idx in deprecated_ids:
            if idx in train_ids:
                train_ids.remove(idx)

        for idx in deprecated_ids:
            if idx in val_ids:
                val_ids.remove(idx)

        self.train_ids = train_ids

        #fold0
        # self.val_ids = [
        #     '114235076',
        #     '193365288',
        #     '324225693',
        #     '398977019',
        #     '1823626595'
        # ]

        self.val_ids = val_ids[:5]

    def setup(self, stage: str = None):
        self.train_dataset = DeformDataset(self.cfg, self.train_ids, True)
        self.val_dataset = DeformDataset(self.cfg, self.val_ids, False)

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