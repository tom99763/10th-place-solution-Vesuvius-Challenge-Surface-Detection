import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageSequence
import shutil
import json
import subprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fft_conv_pytorch")

sys.path.append('./src/nnUNet')

os.environ["nnUNet_raw"] = "../../nnunet/nnUNet_raw_data_base/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "../../nnunet/preprocessed"
os.environ["nnUNet_results"] = "../../nnunet/nnUNet_results"

#configs
plt.ion()   # interactive mode
SPACING = [1, 1, 1]  # change if needed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_json = {
  "name": "Vesuvius Scroll Surface Detection",
  "description": "Binary segmentation of ink regions in 3D X-ray micro-CT TIFF volumes.",
  "reference": "",
  "licence": "",
  "release": "0.1",
  "tensorImageSize": "3D",

  "channel_names": {
    "0": "CT"
  },
  "labels": {
     "background" : 0,
     "surface" : 1,
     "ignore" : 2
  },
  "file_ending": ".tif",
  "numTraining": 806,
}


def safe_tiff_read(path):
    """Fast multi-page TIFF reader using PIL (handles LZW)."""
    img = Image.open(path)

    # read all pages efficiently
    frames = [np.array(frame) for frame in ImageSequence.Iterator(img)]

    # stack into (Z, H, W) or (Z, H, W, C)
    return np.stack(frames, axis=0)


def run_cmd(cmd_list):
    print("\n>>> Running:", " ".join(cmd_list))
    result = subprocess.run(cmd_list, check=True)
    print(">>> Done.\n")
    return result



def main():
    # --- INPUT DATA ---
    DATA_DIR = Path("../../data/vesuvius-challenge-surface-detection")
    CSV_PATH = DATA_DIR / "train.csv"
    IMG_DIR = DATA_DIR / "train_images"
    LBL_DIR = DATA_DIR / "train_labels"
    REPO_DIR = 'nnunet_repo'
    # --- OUTPUT NNUNET DIR ---
    BASE = Path("../../nnunet/nnUNet_raw_data_base/nnUNet_raw")
    task_id = 900
    task_name = "VesuviusScroll"
    task_folder = f"Dataset{task_id:03d}_{task_name}"

    base_dir = BASE / task_folder
    imagesTr = base_dir / "imagesTr"
    labelsTr = base_dir / "labelsTr"

    # make folders
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying TIFFs"):
        case_id = str(row["id"])
        scroll_id = str(row["scroll_id"])

        img_path = IMG_DIR / f"{case_id}.tif"
        lbl_path = LBL_DIR / f"{case_id}.tif"

        if not img_path.exists() or not lbl_path.exists():
            print(f"Skipping missing pair: {scroll_id}")
            continue

        # destination names for nnUNet
        img_dst = imagesTr / f"{case_id}_0000.tif"
        lbl_dst = labelsTr / f"{case_id}.tif"

        json_dst_img = imagesTr / f"{case_id}.json"
        json_dst_lbl = labelsTr / f"{case_id}.json"

        # ---- COPY FILES (FAST) ----
        shutil.copy2(img_path, img_dst)
        shutil.copy2(lbl_path, lbl_dst)

        # ---- WRITE SPACING JSON ----
        spacing_info = {"spacing": SPACING}

        with open(json_dst_img, "w") as f:
            json.dump(spacing_info, f)

        with open(json_dst_lbl, "w") as f:
            json.dump(spacing_info, f)

    with open(base_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("✔️ dataset.json written to:", base_dir / "dataset.json")

    sys.path.append(REPO_DIR)

    run_cmd([
        sys.executable,
        "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
        "-d", "900",
        "-c", "3d_fullres",
        "--verify_dataset_integrity"
    ])



if __name__ == '__main__':
    main()









