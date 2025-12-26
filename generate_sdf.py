import numpy as np
from src.procs.proc_data import *
from tqdm import tqdm
import pandas as pd
from pathlib import Path

data_path = Path("./data/vesuvius-challenge-surface-detection")
save_path = Path("./data/train_sdf_labels")
save_path.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(data_path / "train.csv")
    df = df[~df["id"].astype(str).isin(deprecated_ids)]

    # ---- sparse code all cases ----
    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_id = row["id"]
        out_file = save_path / f"{case_id}_sdf.npz"
        if out_file.exists():
            continue

        mask = load_volume(data_path / "train_labels" / f"{case_id}.tif")
        mask = mask * (mask != 2)

        mask_ds = downsample_mask(mask.astype(np.float32), 2)
        sdt = compute_sdt(mask_ds)
        np.savez_compressed(out_file, sdt=sdt)


if __name__ == '__main__':
    main()