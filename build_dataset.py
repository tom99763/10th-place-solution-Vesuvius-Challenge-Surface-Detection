import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import tifffile
from src.procs.proc_data import *

data_path = './data/vesuvius-challenge-surface-detection'

def main():
    base_dir = Path(data_path)
    meta_data = pd.read_csv(base_dir / 'train.csv')
    img_dir = base_dir / "train_images"

    mask_dir = Path('./data/rough_masks')
    mask_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(meta_data.iterrows(), total=len(meta_data), desc="Processing volumes"):
        image_id = row["id"]
        tif_mask_path = mask_dir / f"rough_mask_{image_id}.tif"
        if tif_mask_path.exists():
            continue

        img_path = img_dir / f"{image_id}.tif"
        volume = load_volume(img_path)
        _, rough_mask = smarter_predict(volume)

        # Save as TIFF
        tifffile.imwrite(tif_mask_path, rough_mask.astype(np.uint8))


if __name__ == '__main__':
    main()