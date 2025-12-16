import topometrics.leaderboard

import numpy as np

from pathlib import Path
import json
import itk
import csv

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import cc3d
import sys


def tee_print(*args, **kwargs):
    print(*args, **kwargs)  # normal print to console
    with open(log_file, "a") as f:
        print(*args, **kwargs, file=f)  # also write to file


def calc_score(labelarr, predarr):
    score_report = topometrics.leaderboard.compute_leaderboard_score(
        predictions=predarr,
        labels=labelarr,
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),  # (z, y, x)
        surface_tolerance=surface_tolerance,  # in spacing units
        voi_connectivity=voi_connectivity,
        voi_transform=voi_transform,
        voi_alpha=voi_alpha,
        combine_weights=(topo_weight, surface_dice_weight, voi_weight),  # (Topo, SurfaceDice, VOI)
        fg_threshold=None,  # None => legacy "!= 0"; else uses "x > threshold"
        ignore_label=2,  # voxels with this GT label are ignored
        ignore_mask=None,  # or pass an explicit boolean mask
    )

    return score_report


def compute_score_for_id(args):
    """Runs calc_score for a single val_id inside a worker process."""
    val_id, fold_idx = args

    gt = itk.imread(data_dir / f"train_labels/{val_id}.tif")
    pred = itk.imread(nnunet_results / 'oof_deform_gaussian'/ f"fold{fold_idx}/{val_id}.tif")

    gtarr = itk.GetArrayFromImage(gt)
    predarr = itk.GetArrayFromImage(pred)

    #  predarr = cc3d.dust(
    #            predarr,
    #            threshold=16, # min_size
    #            connectivity=6, # equivalent to structure=struct_6 usually
    #            in_place=True   # Modifies array in-place to save RAM
    #        )
    #
    #  # predarr is now labeled. Convert back to boolean/uint8 if needed for scoring
    #  predarr[predarr > 0] = 1
    #
    score_report = calc_score(gtarr, predarr)

    return {
        "val_id": val_id,
        "fold": fold_idx,
        "score": score_report.score,
        "topo_score": score_report.topo.toposcore,
        "voi_score": score_report.voi.voi_score,
        "surface_dice": score_report.surface_dice,
    }


def main():
    with open(nnunet_processed / "splits_final.json", "r") as f:
        fold_splits = json.load(f)

        # open CSV for writing
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["val_id", "fold", "score", "topo_score", "voi_score", "surface_dice"]
        )
        writer.writeheader()

        for fold_idx in range(4):
            val_ids = fold_splits[fold_idx]['val']
            tee_print(f"Processing fold {fold_idx} with {len(val_ids)} validation samples")

            # prepare batch of jobs
            jobs = [(val_id, fold_idx) for val_id in val_ids]

            running = {
                "count": 0,
                "score": 0.0,
                "topo_score": 0.0,
                "voi_score": 0.0,
                "surface_dice": 0.0,
            }

            if not PARALLEL:

                for (val_id, fold_idx) in tqdm(jobs):
                    result = compute_score_for_id((val_id, fold_idx))
                    writer.writerow(result)

                    # Update running totals
                    running["count"] += 1
                    running["score"] += result["score"]
                    running["topo_score"] += result["topo_score"]
                    running["voi_score"] += result["voi_score"]
                    running["surface_dice"] += result["surface_dice"]

                    # Print online stats every 10 samples
                    if running["count"] % 10 == 0:
                        tee_print(
                            f"[Fold {fold_idx}] Processed {running['count']} samples "
                            f"| Avg score: {running['score'] / running['count']:.4f}, "
                            f"Avg topo: {running['topo_score'] / running['count']:.4f}, "
                            f"Avg VOI: {running['voi_score'] / running['count']:.4f}, "
                            f"Avg surface dice: {running['surface_dice'] / running['count']:.4f}"
                        )
            else:

                with ProcessPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(compute_score_for_id, job): job for job in jobs}

                    for future in tqdm(as_completed(futures), total=len(futures)):
                        result = future.result()
                        writer.writerow(result)

                        # Update running totals
                        running["count"] += 1
                        running["score"] += result["score"]
                        running["topo_score"] += result["topo_score"]
                        running["voi_score"] += result["voi_score"]
                        running["surface_dice"] += result["surface_dice"]

                        # Print online stats every 10 samples
                        if running["count"] % 10 == 0:
                            tee_print(
                                f"[Fold {fold_idx}] Processed {running['count']} samples "
                                f"| Avg score: {running['score'] / running['count']:.4f}, "
                                f"Avg topo: {running['topo_score'] / running['count']:.4f}, "
                                f"Avg VOI: {running['voi_score'] / running['count']:.4f}, "
                                f"Avg surface dice: {running['surface_dice'] / running['count']:.4f}"
                            )

            tee_print(f"Finished fold {fold_idx}")

    tee_print(f"\nAll folds completed. CSV saved to: {output_csv}")

    tee_print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    log_file = sys.argv[1]
    output_csv = "oof_scores.csv"
    data_dir = Path("./data/vesuvius-challenge-surface-detection")
    nnunet_raw = Path("./nnunet/nnUNet_raw_data_base/nnUNet_raw/Dataset900_VesuviusScroll")
    nnunet_processed = Path("./nnunet/preprocessed/Dataset900_VesuviusScroll")
    nnunet_results = Path(
        "./nnunet/nnUNet_results/Dataset900_VesuviusScroll/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")

    # Define structure for faces only (Manhattan distance = 1)
    struct_6 = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    # COMP METRICS PARAMS
    surface_tolerance: float = 2.0
    voi_connectivity: int = 26
    voi_transform: str = 'one_over_one_plus'
    voi_alpha: float = 0.3
    topo_weight: float = 0.3
    surface_dice_weight: float = 0.35
    voi_weight: float = 0.35

    PARALLEL = True
    main()