import numpy as np
from PIL import Image, ImageSequence
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label
from scipy import ndimage as ndi


def load_volume(path: Path) -> np.ndarray:
    """
    Load a multi-page TIFF into a 3D NumPy array: (slices, H, W)
    """
    try:
        with Image.open(path) as img:
            frames = [np.array(frame) for frame in ImageSequence.Iterator(img)]
        volume = np.stack(frames)
        return volume
    except Exception as e:
        raise RuntimeError(f"Error loading TIFF {path}: {e}")


def get_rough_mask(mask):
    mask_closed = ndi.binary_closing(mask, structure=np.ones((5 ,5 ,5)))
    mask_dilated = ndi.binary_dilation(mask_closed, iterations=3)
    mask_filled = remove_small_holes(mask_dilated, area_threshold=200)
    return mask_filled


def smarter_predict(volume: np.ndarray):
    """
    A non-ML baseline using Otsu's threshold
    followed by morphological cleanup.
    """
    try:
        thresh = threshold_otsu(volume)
        mask = (volume > thresh)
    except ValueError:

        print("     Otsu thresholding failed, falling back to mean.")
        mean_val = volume.mean()
        mask = (volume > mean_val)

    labeled_mask = label(mask)
    cleaned_mask = remove_small_objects(labeled_mask, min_size=5000)

    final_mask = (cleaned_mask > 0).astype(np.uint8)

    rough_mask = get_rough_mask(final_mask)

    return final_mask, rough_mask