import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
import nibabel as nib  # optional, for NIfTI I/O

def mask_to_sdf(mask: np.ndarray, max_dist: float = 20.0, ignore_label: int = None) -> np.ndarray:
    """
    Convert a binary mask to a Signed Distance Field (SDF).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, shape (D, H, W) or (1, D, H, W)
        1 = object, 0 = background
    max_dist : float
        Maximum distance to clip the SDF
    ignore_label : int or None
        If specified, voxels with this value are ignored and set to max_dist + 1

    Returns
    -------
    sdf : np.ndarray
        Signed distance field, same shape as input mask (without channel dim)
        Inside object: negative, outside: positive, surface: 0
    """
    if mask.ndim == 4:
        mask = mask[0]  # remove channel dim if present

    mask_bin = mask.astype(np.uint8)

    # Optional ignore label
    if ignore_label is not None:
        ignore = mask_bin == ignore_label
    else:
        ignore = None

    fg = mask_bin > 0  # foreground
    dist_out = distance_transform_edt(~fg)  # outside distance
    dist_in = distance_transform_edt(fg)    # inside distance

    sdf = dist_out - dist_in

    # Clip extreme values
    sdf = np.clip(sdf, -max_dist, max_dist)

    # Mark ignore voxels if needed
    if ignore is not None:
        sdf[ignore] = max_dist + 1.0

    return sdf.astype(np.float32)


# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    # Load a NIfTI mask (optional)
    mask_nii = nib.load("mask.nii.gz")
    mask_np = mask_nii.get_fdata()

    # Convert to SDF
    sdf_np = mask_to_sdf(mask_np, max_dist=20.0)

    # Save as NIfTI
    sdf_nii = nib.Nifti1Image(sdf_np, mask_nii.affine)
    nib.save(sdf_nii, "mask_sdf.nii.gz")

    print("SDF saved to mask_sdf.nii.gz")