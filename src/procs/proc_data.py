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
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import zoom

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

def upsample_mask(mask: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return mask
    x = mask[None, None]
    x = F.interpolate(x, scale_factor=factor, mode="nearest")
    return x[0, 0]

def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask
    x = torch.tensor(mask, dtype=torch.float32)[None, None]
    x = F.max_pool3d(x, factor, factor)
    return x[0, 0].numpy()

def reconstruct_mask(Z, D, upsample_factor=2):
    pad = D.shape[2] // 2
    sdf = F.conv3d(Z, D, padding=pad, groups=D.shape[0]).sum(1)
    mask = (sdf[0] >= 0).float()
    return upsample_mask(mask, upsample_factor)


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


def load_sparse_tensor(path: str) -> np.ndarray:
    """
    Load a sparse tensor saved in index+value+shape format
    Returns dense np.ndarray
    """
    data = np.load(path)
    tensor = np.zeros(data['shape'], dtype=np.float32)
    tensor[data['idx']] = data['values']
    return tensor


def compute_sdt(mask: np.ndarray, clip: float = 20.0) -> np.ndarray:
    sdt = distance_transform_edt(mask) - distance_transform_edt(1 - mask)
    return np.clip(sdt, -clip, clip)


def downsample_volume_np(vol, factor=2):
    return zoom(vol, zoom=1/factor, order=1)


deprecated_ids = [
    "1407735",
    "1641318781",
    "2268221981",
    "2996066940",
    "3241425466",
    "808135176",
    "1924200298",
    "2376256768",
    "3012554500",
    "3398456664",
    "862434992",
    "1951193117",
    "2573842867",
    "3110212378",
    "3470951309",
    "885379642",
    "2075542469",
    "2791137336",
    "3215721649",
    "3518241476"
]



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

def load_sparse_z(path: Path, device=None):
    """Load sparse .npz and return dense tensor on given device."""
    d = np.load(path)
    idx = torch.tensor(d["idx"], dtype=torch.long)
    vals = torch.tensor(d["vals"], dtype=torch.float32)
    shape = tuple(d["shape"])

    D = torch.zeros(shape, dtype=vals.dtype, requires_grad=False)
    D[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = vals
    if device:
        D = D.to(device)
    return D


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
