from src.procs.proc_data import *
from pathlib import Path
import os
from tqdm import tqdm


def main():
    oof_path = Path('data/nnunet_oof')
    if not os.path.exists(oof_path/'padded'):
        os.makedirs(oof_path/'padded')
    names = os.listdir(oof_path/'oof')
    for name in tqdm(names):
        path = oof_path/'oof'/name
        mask = load_volume(path)
        mask = pad_skeleton_3d(mask)
        np.save(oof_path/'padded'/f'{name.split('.tif')[0]}.npy', mask)

if __name__ == '__main__':
    main()