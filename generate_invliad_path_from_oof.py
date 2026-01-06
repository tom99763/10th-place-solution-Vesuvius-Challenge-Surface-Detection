from src.procs.proc_data import *
import os
from tqdm import tqdm
from pathlib import Path


def main():
    data_path = Path('data/vesuvius-challenge-surface-detection')
    oof_path = Path('data/nnunet_oof')
    if not os.path.exists(oof_path/'invalid_labels'):
        os.makedirs(oof_path/'invalid_labels')
    names = os.listdir(oof_path/'oof')
    for name in tqdm(names):
        pred_path = oof_path/'oof'/name
        gt_path = data_path/'train_labels'/name
        pred_mask = load_volume(pred_path)
        gt_mask = load_volume(gt_path)
        D, H, W = pred_mask.shape
        output_mask = np.zeros_like(pred_mask)
        for j in range(D):
            invalid_mask, _, _, _ = filter_cc_by_gt_ratio(pred_mask[j], gt_mask[j])
        np.save(oof_path/'invalid_labels'/f'{name.split('.tif')[0]}.npy', output_mask)

if __name__ == '__main__':
    main()