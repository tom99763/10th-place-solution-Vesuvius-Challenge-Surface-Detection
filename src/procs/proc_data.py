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
from skimage.morphology import binary_dilation, square
from skimage.morphology import remove_small_objects
import cc3d

logger = logging.getLogger(__name__)

def pad_skeleton_2d(skel_2d, pad=1):
    """
    skel_2d: (H, W) binary skeleton
    pad=1 → 3x3 padding
    """
    selem = square(2 * pad + 1)
    return binary_dilation(skel_2d, selem)


def pad_skeleton_3d(mask, pad = 1):
    D, H, W = mask.shape
    output_mask = np.zeros_like(mask)
    for i in range(D):
        padded_skel = pad_skeleton_2d(mask[i], pad)
        output_mask[i] = padded_skel
    return output_mask


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
    d = np.load(path, mmap_mode='r')
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



def filter_cc_by_gt_ratio(
    pred_2d: np.ndarray,
    gt_2d: np.ndarray,
    tau: float = 0.001,
    connectivity: int = 8,
    thr: float = 0.5,
):
    """
    For each connected component in pred_2d, compute:
        ratio = (# GT==1 inside the component) / (component size)

    If ratio > tau -> remove component
    Else           -> keep component

    Args:
        pred_2d: (H, W) prediction (float or binary)
        gt_2d:   (H, W) ground truth binary mask
        tau:     threshold in [0, 1]
        connectivity: 4 or 8 for 2D CC
        thr:     threshold to binarize pred_2d

    Returns:
        filtered_mask: (H, W) binary mask after filtering
        kept_labels:   list of kept CC ids
        removed_labels:list of removed CC ids
        ratios:        dict {cc_id: ratio}
    """

    # --- binarize ---
    pred_bin = (pred_2d > thr).astype(np.uint8)
    gt_bin = (gt_2d > 0).astype(np.uint8)

    # --- connected components ---
    labels, num_cc = cc3d.connected_components(
        pred_bin,
        connectivity=connectivity,
        return_N=True
    )

    filtered_mask = np.zeros_like(pred_bin, dtype=np.uint8)
    kept_labels, removed_labels = [], []
    ratios = {}

    # --- per-component processing ---
    for k in range(1, num_cc + 1):
        cc_mask = (labels == k)
        cc_size = cc_mask.sum()

        if cc_size == 0:
            continue

        # GT ratio *inside* the component
        gt_inside = gt_bin[cc_mask].sum()
        ratio = gt_inside / cc_size
        ratios[k] = ratio

        if ratio > tau:
            removed_labels.append(k)
        else:
            filtered_mask[cc_mask] = 1
            kept_labels.append(k)

    return filtered_mask, kept_labels, removed_labels, ratios

# https://www.kaggle.com/code/choudharymanas/inference-baseline-transunet-lb-0-537
def build_anisotropic_struct(z_radius: int, xy_radius: int):
    z, r = z_radius, xy_radius
    if z == 0 and r == 0:
        return None
    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct
    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct
    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct


def mask_to_sdf(mask_patch, clip_distance=20.0):
    """
    Convert a binary mask patch to normalized SDF in [-1, 1].

    mask_patch: np.ndarray of shape (D,H,W), values 0 (bg), 1 (fg)
    clip_distance: max distance to clip (in voxels)
    """
    # Foreground SDF
    fg_mask = mask_patch == 1
    bg_mask = mask_patch == 0

    # EDT distances
    sdf_fg = distance_transform_edt(fg_mask)
    sdf_bg = distance_transform_edt(bg_mask)

    # Signed distance
    sdf = sdf_bg - sdf_fg  # negative inside fg, positive outside

    # Clip & normalize
    sdf = np.clip(sdf, -clip_distance, clip_distance) / clip_distance
    sdf = sdf.astype(np.float32)
    return sdf


def mask_to_sdf_parallel(masks, clip_distance=20.0):
    """
    Convert a batch of masks (B, 1, D, H, W) or (B, D, H, W) to SDF in [-1, 1].
    Handles unlabeled (2) as background.
    """
    # convert to numpy
    is_torch = isinstance(masks, torch.Tensor)
    if is_torch:
        masks = masks.cpu().numpy()

    if masks.ndim == 4:  # (B, D, H, W) -> (B, 1, D, H, W)
        masks = masks[:, None, ...]

    B, C, D, H, W = masks.shape
    masks = masks.astype(np.uint8)
    masks[masks == 2] = 0  # treat unlabeled as background

    # Prepare arrays
    sdf_batch = np.zeros_like(masks, dtype=np.float32)

    # vectorized computation using list comprehension (still parallelizable)
    fg_masks = masks[:, 0] == 1
    bg_masks = masks[:, 0] == 0

    # distance transform
    sdf_fg_list = [distance_transform_edt(fg_masks[b]) for b in range(B)]
    sdf_bg_list = [distance_transform_edt(bg_masks[b]) for b in range(B)]

    # stack
    sdf_fg = np.stack(sdf_fg_list, axis=0)
    sdf_bg = np.stack(sdf_bg_list, axis=0)

    sdf = sdf_bg - sdf_fg
    sdf = np.clip(sdf, -clip_distance, clip_distance) / clip_distance
    sdf_batch[:, 0] = sdf

    if is_torch:
        sdf_batch = torch.from_numpy(sdf_batch)

    return sdf_batch

def topo_postprocess(
    probs,
    T_low=0.30,
    T_high=0.80,
    z_radius=3,
    xy_radius=2,
    dust_min_size=100,
):
    # Step 1: 3D Hysteresis
    strong = probs >= T_high
    weak   = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(
        strong, mask=weak, structure=struct_hyst
    )

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Step 2: 3D Anisotropic Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: Dust Removal
    if dust_min_size > 0:
        mask = remove_small_objects(
            mask.astype(bool), min_size=dust_min_size
        )

    return mask.astype(np.uint8)
