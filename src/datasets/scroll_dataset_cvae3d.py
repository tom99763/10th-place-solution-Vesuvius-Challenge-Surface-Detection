from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from ..procs.proc_data import *
from pathlib import Path
from src.procs.proc_utils import *
import torch.nn.functional as F


class CVAEDataset(Dataset):
    def __init__(self, opt, is_train):
        super().__init__()
        self.opt = opt
        self.is_train = is_train
        self.id_list = opt.train_val_splits_ids[opt.selected_fold]['train'] if is_train\
            else opt.train_val_splits_ids[opt.selected_fold]['val']

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        idx = self.id_list[idx]
        mask_gt = load_volume(self.opt.dataset_dir/'train_labels'/f'{idx}.tif')
        mask_pred = load_volume(self.opt.oof_dir/f'{idx}.tif')
        mask_gt = torch.from_numpy(mask_gt).float()
        mask_pred = torch.from_numpy(mask_pred).float()
        mask_gt = mask_gt.unsqueeze(0).unsqueeze(0)
        mask_pred = mask_pred.unsqueeze(0).unsqueeze(0)
        mask_gt = F.interpolate(mask_gt, size=self.opt.scaled_size, mode="nearest")
        mask_pred = F.interpolate(mask_pred, size=self.opt.scaled_size, mode="nearest")
        mask_gt = mask_gt.squeeze(0).squeeze(0)
        mask_pred = mask_pred.squeeze(0).squeeze(0)
        return mask_gt, mask_pred

    def setup_scale(self, scale_idx):
        size = get_scales_by_index(scale_idx, self.opt.scale_factor, self.opt.stop_scale, self.opt.img_size)
        self.opt.scaled_size = [size] * 3
