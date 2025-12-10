import numpy as np
from PIL import Image, ImageSequence
from pathlib import Path
# from skimage.filters import threshold_otsu
# from skimage.morphology import remove_small_objects, remove_small_holes
# from skimage.measure import label
from scipy import ndimage as ndi
import logging
from monai import transforms
from monai.transforms import Compose
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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


def gaussian_kernel_3d(kernel_size=5, sigma=1.0, device="cuda"):
    """Returns a normalized 3D Gaussian kernel (1,1,K,K,K)."""
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_3d(x, kernel_size=7, sigma=10.0):
    """
    x: (B, C, D, H, W)
    """
    B, C, D, H, W = x.shape
    kernel = gaussian_kernel_3d(kernel_size, sigma, device=x.device)

    # shape: (C, 1, K, K, K)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size, kernel_size)

    # depthwise convolution
    return F.conv3d(x, kernel, padding=kernel_size // 2, groups=C)



# def get_rough_mask(mask):
#     mask_closed = ndi.binary_closing(mask, structure=np.ones((5 ,5 ,5)))
#     mask_dilated = ndi.binary_dilation(mask_closed, iterations=3)
#     mask_filled = remove_small_holes(mask_dilated, area_threshold=200)
#     return mask_filled


# def smarter_predict(volume: np.ndarray):
#     """
#     A non-ML baseline using Otsu's threshold
#     followed by morphological cleanup.
#     """
#     try:
#         thresh = threshold_otsu(volume)
#         mask = (volume > thresh)
#     except ValueError:
#
#         print("     Otsu thresholding failed, falling back to mean.")
#         mean_val = volume.mean()
#         mask = (volume > mean_val)
#
#     labeled_mask = label(mask)
#     cleaned_mask = remove_small_objects(labeled_mask, min_size=5000)
#
#     final_mask = (cleaned_mask > 0).astype(np.uint8)
#
#     rough_mask = get_rough_mask(final_mask)
#
#     return final_mask, rough_mask


def generate_transforms(
    transforms_config: list[dict],
) -> list[transforms.Transform]:
    transform_list = []
    logger.debug(f"Generating {len(transforms_config)} transforms")

    for transform_config in transforms_config:
        transform_name = next(iter(transform_config))
        transform_kwargs = transform_config[transform_name]
        logger.debug(
            f"Generating transform {transform_name} with kwargs {transform_kwargs}"
        )
        transform: transforms.Transform = getattr(transforms, transform_name)(
            **transform_kwargs
        )  # type: ignore
        transform_list.append(transform)
    return Compose(transform_list)  # type: ignore
