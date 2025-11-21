import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.procs.proc_data import *

data_path = '/kaggle/data/'
neighb_width = 1  # for 3 channels

def extract_with_padding(volume, idx, neighb_width):
    """
    Returns a (2*neighb_width+1, H, W) slice stack.
    Edges are handled by repeating boundary slices.
    """
    depth = volume.shape[0]
    required = 2 * neighb_width + 1

    # Initial slice indices
    lo = idx - neighb_width
    hi = idx + neighb_width + 1

    # Extract valid part
    vol_part = volume[max(lo, 0):min(hi, depth)]

    # Left padding (repeat first available slice)
    if lo < 0:
        left_pad = np.repeat(volume[[0]], -lo, axis=0)
    else:
        left_pad = None

    # Right padding (repeat last available slice)
    if hi > depth:
        right_pad = np.repeat(volume[[depth - 1]], hi - depth, axis=0)
    else:
        right_pad = None

    # Concatenate all parts
    parts = []
    if left_pad is not None:
        parts.append(left_pad)
    parts.append(vol_part)
    if right_pad is not None:
        parts.append(right_pad)

    stacked = np.concatenate(parts, axis=0)

    # Final sanity check
    assert stacked.shape[0] == required, f"Wrong slice count: {stacked.shape}"

    return stacked


def main():
    base_dir = Path(data_path)
    meta_data = pd.read_csv(base_dir / 'train.csv')
    img_dir = base_dir / "train_images"
    mask_dir = base_dir / "train_labels"

    out_img_root = Path(f'./data/train_images_25d_{neighb_width}')
    out_mask_root = Path(f'./data/train_labels_25d_{neighb_width}')
    out_img_root.mkdir(parents=True, exist_ok=True)
    out_mask_root.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(meta_data.iterrows(), total=len(meta_data), desc="Processing volumes"):
        image_id = row["id"]

        img_path = img_dir / f"{image_id}.tif"
        mask_path = mask_dir / f"{image_id}.tif"

        volume = load_volume(img_path)
        mask = load_volume(mask_path)

        depth = volume.shape[0]

        img25d_dir = out_img_root / str(image_id)
        img25d_dir.mkdir(parents=True, exist_ok=True)

        mask25d_dir = out_mask_root / str(image_id)
        mask25d_dir.mkdir(parents=True, exist_ok=True)

        for i in range(depth):
            img25d = extract_with_padding(volume, i, neighb_width)
            mask25d = extract_with_padding(mask, i, neighb_width)

            np.save(img25d_dir / f'img25d_{i}.npy', img25d)
            np.save(mask25d_dir / f'mask25d_{i}.npy', mask25d)

if __name__ == '__main__':
    main()
