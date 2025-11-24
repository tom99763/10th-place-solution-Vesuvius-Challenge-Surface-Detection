import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from src.procs.proc_data import generate_transforms
from tqdm import tqdm

# Function to pad channels to 3
def pad_channels_to_3(arr):
    c, h, w = arr.shape
    if c < 3:
        pad = np.zeros((3 - c, h, w), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


class ScrollDataset25D(Dataset):
    """
    2.5D dataset. Supports preloading validation data into RAM for faster validation.
    """

    def __init__(self, cfg, id_list, mode="train", preload=False):
        super().__init__()
        self.cfg = cfg
        self.id_list = id_list
        self.mode = mode
        self.preload = preload

        # transforms
        if mode == 'train':
            self.proc_vol_and_mask = generate_transforms(cfg.data.transforms.train)
        else:
            self.proc_vol_and_mask = generate_transforms(cfg.data.transforms.val)

        # Preload data into memory if requested
        if self.preload:
            self.vols = []
            self.masks = []
            print('preload vol and mask...')
            for slice_path in tqdm(self.id_list):
                vol, mask = self._load_slice(slice_path)
                self.vols.append(vol)
                self.masks.append(mask)

    def __len__(self):
        return len(self.id_list)

    def _load_slice(self, slice_path):
        # Load numpy arrays fully into memory
        vol = np.load(slice_path).astype(np.float32)
        mask_path = Path(str(slice_path).replace("train_images_25d_1", "train_labels_25d_1"))
        mask_path = Path(str(mask_path).replace("img", "mask"))
        mask = np.load(mask_path).astype(np.float32)

        # Ensure channel-first shape
        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
        vol = pad_channels_to_3(vol)

        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]
        mask = pad_channels_to_3(mask)

        return vol, mask

    def __getitem__(self, idx):
        if self.preload:
            vol = self.vols[idx]
            mask = self.masks[idx]
        else:
            slice_path = self.id_list[idx]
            vol, mask = self._load_slice(slice_path)

        transformed = self.proc_vol_and_mask({"Image": vol, "Mask": mask})
        return transformed["Image"], transformed["Mask"]


class ScrollDataModule25D(pl.LightningDataModule):
    def __init__(self, cfg, train_ids, val_ids):
        super().__init__()
        self.cfg = cfg
        self.train_ids = train_ids
        self.val_ids = val_ids

    def setup(self, stage=None):
        # Train dataset: lazy-loaded
        self.train_dataset = ScrollDataset25D(
            self.cfg, self.train_ids, mode="train", preload=False
        )

        # Validation dataset: preload into RAM
        self.val_dataset = ScrollDataset25D(
            self.cfg, self.val_ids, mode="val", preload=True
        )

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