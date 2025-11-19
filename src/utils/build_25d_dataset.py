import pandas as pd
import os
from pathlib import Path
import numpy as np
import tifffile
from src.procs.proc_data import *
from tqdm import tqdm

data_path = '/kaggle/data/'
neighb_width = 1

def main():
    base_dir = Path(data_path)
    meta_data = pd.read_csv(base_dir / 'train.csv')
    img_dir = base_dir / "train_images"
    mask_dir = base_dir / "train_labels"

    file25d_dir = Path(f'./data/train_images_25d_{neighb_width}')
    file25d_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(meta_data.iterrows(), total=len(meta_data), desc="Processing volumes"):
        image_id = row["id"]
        img_path = img_dir / f"{image_id}.tif"
        mask_path = mask_dir / f"{image_id}.tif"
        volume = load_volume(img_path)
        mask = load_volume(mask_path)

        img25d_dir = Path(f'./data/train_images_25d_{neighb_width}/{image_id}')
        img25d_dir.mkdir(parents=True, exist_ok=True)

        mask25d_dir = Path(f'./data/train_labels_25d_{neighb_width}/{image_id}')
        mask25d_dir.mkdir(parents=True, exist_ok=True)

        depth = volume.shape[0]
        for i in range(neighb_width, depth):
            img25d = volume[i-neighb_width:i+neighb_width+1]
            mask25d = mask[i-neighb_width:i+neighb_width+1]
            np.save(img25d_dir/f'img25d_{i}.npy', img25d)
            np.save(mask25d_dir / f'mask25d_{i}.npy', mask25d)

if __name__ == '__main__':
    main()