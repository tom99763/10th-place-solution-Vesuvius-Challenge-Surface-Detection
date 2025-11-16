import os
import pandas as pd
from tqdm import tqdm
from src.procs.proc_data import *

data_path = './data/vesuvius-challenge-surface-detection'


def main():
    base_dir = Path(data_path)
    meta_data = pd.read_csv(base_dir/'train.csv')
    img_dir = base_dir / "train_images"

    if not os.path.exists('./data/rough_masks'):
        os.makedirs('./data/rough_masks')

    for _, row in tqdm(meta_data.iterrows(), total=len(meta_data), desc="Processing volumes"):
        image_id = row["id"]
        if os.path.exists(f"./data/rough_masks/rough_mask_{image_id}.npz"):
            continue
        filename = f"{image_id}.tif"
        img_path = img_dir / filename
        volume = load_volume(img_path)
        _, rough_mask = smarter_predict(volume)
        np.savez_compressed(f"./data/rough_masks/rough_mask_{image_id}.npz", rough_mask=rough_mask)

if __name__ == '__main__':
    main()