import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.procs.proc_data import load_volume
from src.procs.decompose import ExactScrollDecomposition

data_path = Path('./data/vesuvius-challenge-surface-detection')
save_path = Path('./data/train_compose_labels_npz')


def main():
    save_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path / 'train.csv')
    decomp = ExactScrollDecomposition()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generate Compressed NPZs"):
        case_id = row['id']
        mask_path = data_path / 'train_labels' / f'{case_id}.tif'
        mask = load_volume(mask_path)
        mask = mask * (mask != 2)
        components = decomp.transform(mask)

        # Save all components in a single compressed npz
        np.savez_compressed(
            save_path / f"{case_id}.npz",
            thickness=components['thickness'],
            sdf=components['sdf'],
            normals=components['normals']
        )

if __name__ == '__main__':
    main()