import os
import numpy as np
from src.procs.proc_data import *
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import binary_erosion
from skimage.morphology import skeletonize
import tifffile as tiff

data_path = Path('./data/vesuvius-challenge-surface-detection')
save_path = Path('./data/train_compose_labels')

def main():
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.read_csv(data_path/'train.csv')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generate Compose TIFFs"):
        case_id = row.id
        mask_path = data_path/'train_labels'/f'{case_id}.tif'
        mask = load_volume(mask_path)
        mask = mask * (mask!=2)
        depth = mask.shape[0]
        skeleton3d = np.zeros_like(mask)
        edge3d = np.zeros_like(mask)
        cover3d = np.zeros_like(mask)
        for i in range(depth):
            mask2d = mask[i]
            skeleton = skeletonize(mask2d).astype('float32')
            edge = (mask2d ^ binary_erosion(mask2d)).astype('float32')
            cover = np.maximum(mask2d.astype('float32') - skeleton - edge, 0)
            skeleton3d[i] = skeleton
            edge3d[i] = edge
            cover3d[i] = cover
        tiff.imwrite(save_path/ "skeleton3d.tif", skeleton3d.astype(np.uint8))
        tiff.imwrite(save_path/"edge3d.tif", edge3d.astype(np.uint8))
        tiff.imwrite(save_path/"cover3d.tif", cover3d.astype(np.uint8))

if __name__ == '__main__':
    main()