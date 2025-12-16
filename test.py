import json
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


def cleanup():
    with open('./splits_final.json', 'r') as f:
        val_splits = json.load(f)
    val_ids = val_splits[4]['val']
    path = Path('nnunet/nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll/imagesTrfold4')
    for name in tqdm(os.listdir(path)):
        id1 = name.split('.')[0]
        id2 = name.split('_')[0]
        if (id1 not in val_ids) and (id2 not in val_ids):
            os.remove(path/name)

if __name__ == '__main__':
    cleanup()