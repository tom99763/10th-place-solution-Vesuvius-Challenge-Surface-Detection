import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tifffile
from src.procs.proc_data import load_volume
from src.procs.decompose import ExactScrollDecomposition

data_path = Path('./data/vesuvius-challenge-surface-detection')
save_path = Path('./data/train_compose_labels')

def main():
    save_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path/'train.csv')
    decomp = ExactScrollDecomposition()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generate Compose TIFFs"):
        case_id = row['id']
        mask_path = data_path/'train_labels'/f'{case_id}.tif'
        mask = load_volume(mask_path)
        mask = mask * (mask != 2)
        components = decomp.transform(mask)
        tifffile.imwrite(save_path/f"thickness_{case_id}.tif", components['thickness'])
        tifffile.imwrite(save_path/f"sdf_{case_id}.tif", components['sdf'])
        tifffile.imwrite(save_path/f"normals_{case_id}.tif", components['normals'])

if __name__ == '__main__':
    main()