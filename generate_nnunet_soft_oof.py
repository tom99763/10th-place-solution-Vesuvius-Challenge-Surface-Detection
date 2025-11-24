import json
from pathlib import Path
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm

base = Path(f"/path/to/nnunetv2/RESULTS_FOLDER/nnUNet/{cfg}/Dataset{dataset_id}")

# load fold splits
with open(f"./nnunet/preprocessed/Dataset900_VesuviusScroll/splits_final.json") as f:
    splits = json.load(f)

for fold, split in enumerate(splits):
    val_cases = split['val']
    predictor = nnUNetPredictor(tile_step_size=0.5, use_mirroring=True)

    predictor.initialize_from_trained_model_folder(
        base / f"fold_{fold}",
        use_folds=(fold,),  # VERY IMPORTANT: only this fold
    )

    for case_name in val_cases:
        pred, _ = predictor.predict_from_files(
            {"image": f"./nnunet/nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll/imagesTr/{case_name}_0000.nii.gz"},
            return_softmax=True,
            save_probabilities=False,
        )

        np.save(f"oof_softmax/{case_name}_soft_fold{fold}.npy", pred)