import os
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from src.procs.proc_data import *
from tqdm import tqdm


def main():
    data_path = Path('./data/vesuvius-challenge-surface-detection')
    filenames = os.listdir(data_path/'new_labels')
    if not os.path.exists(data_path/'new_labels_sdf'):
        os.makedirs(data_path/'new_labels_sdf')
    filenames = [filename for filename in filenames if "Zone" not in filename]
    for filename in tqdm(filenames):
        mask = load_volume(data_path/'new_labels'/filename)
        sdf = mask_to_sdf(mask)
        name = filename.split('.')[0]
        np.save(data_path/'new_labels_sdf'/f'{name}.npy', sdf.astype('float16'))

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    main()