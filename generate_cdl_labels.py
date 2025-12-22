import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from src.procs.proc_data import load_volume
# from cv import calc_score   # optional

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ============================================================
# Determinism
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Resampling
# ============================================================

def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask
    x = torch.tensor(mask, dtype=torch.float32)[None, None]
    x = F.max_pool3d(x, factor, factor)
    return x[0, 0].numpy()


def upsample_mask(mask: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return mask
    x = mask[None, None]
    x = F.interpolate(x, scale_factor=factor, mode="nearest")
    return x[0, 0]


# ============================================================
# Signed Distance Transform
# ============================================================

def compute_sdt(mask: np.ndarray, clip: float = 20.0) -> np.ndarray:
    sdt = distance_transform_edt(mask) - distance_transform_edt(1 - mask)
    return np.clip(sdt, -clip, clip)


# ============================================================
# Dictionary Module
# ============================================================

class ConvDictionary(nn.Module):
    def __init__(self, K: int, k: int, device: str, seed: int):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(seed)
        self.D = nn.Parameter(
            torch.randn(K, 1, k, k, k, generator=g, device=device)
        )
        self.normalize()

    def normalize(self):
        with torch.no_grad():
            n = torch.norm(self.D.flatten(1), dim=1)
            self.D.div_(n.view(-1, 1, 1, 1, 1) + 1e-8)


# ============================================================
# Sparse Z utilities (CRITICAL)
# ============================================================

def dense_to_sparse(Z: torch.Tensor, threshold: float):
    """
    Z: (1, K, D, H, W)
    returns:
        idx: (N, 4) -> (k, z, y, x)
        val: (N,)
        shape: (K, D, H, W)
    """
    mask = Z.abs() > threshold
    idx = mask.nonzero(as_tuple=False)
    vals = Z[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3], idx[:, 4]]
    return idx[:, 1:], vals, Z.shape[1:]


def save_sparse_z(path: Path, idx, vals, shape):
    np.savez_compressed(
        path,
        idx=idx.cpu().numpy(),
        vals=vals.detach().cpu().numpy(),
        shape=np.array(shape),
    )


def load_sparse_z(path: Path, device):
    d = np.load(path)
    return (
        torch.tensor(d["idx"], device=device),
        torch.tensor(d["vals"], device=device),
        tuple(d["shape"]),
    )


# ============================================================
# CDL TRAINING (DICTIONARY ONLY)
# ============================================================

def train_dictionary(
    mask: np.ndarray,
    device="cuda",
    K=64,
    kernel_size=18,
    downsample_factor=2,
    n_epochs=200,
    lambda_sparse=0.6,
    lambda_D=0.01,
    threshold_init=0.01,
    threshold_final=0.02,
    seed=42,
):
    set_seed(seed)

    mask_ds = downsample_mask(mask.astype(np.float32), downsample_factor)
    sdt = compute_sdt(mask_ds)
    X = torch.tensor(sdt, device=device, dtype=torch.float32)[None, None]

    k = max(3, kernel_size // downsample_factor)
    if k % 2 == 0:
        k += 1
    pad = k // 2

    D = ConvDictionary(K, k, device, seed)
    Z = nn.Parameter(torch.randn(1, K, *X.shape[2:], device=device) * 0.01)
    print(f'Dictionary size: {D.D.detach().shape} -- Z size: {Z.shape}')

    opt_D = torch.optim.Adam([D.D], lr=1e-3)
    opt_Z = torch.optim.Adam([Z], lr=1e-2)

    for ep in tqdm(range(n_epochs), desc="CDL Dictionary"):
        thr = threshold_init + (threshold_final - threshold_init) * ep / n_epochs

        # ---- Z update ----
        for _ in range(10):
            opt_Z.zero_grad()
            X_hat = F.conv3d(Z, D.D, padding=pad, groups=K).sum(1, keepdim=True)
            loss = 0.5 * F.mse_loss(X_hat, X) + lambda_sparse * Z.abs().mean()
            loss.backward()
            opt_Z.step()

            with torch.no_grad():
                Z.copy_(torch.sign(Z) * F.relu(Z.abs() - thr))

        # ---- D update ----
        opt_D.zero_grad()
        X_hat = F.conv3d(Z, D.D, padding=pad, groups=K).sum(1, keepdim=True)
        loss_D = 0.5 * F.mse_loss(X_hat, X) + lambda_D * D.D.abs().mean()
        loss_D.backward()
        opt_D.step()
        D.normalize()

    with torch.no_grad():
        D.D[D.D.abs() < threshold_final] = 0

    return D.D.detach()


# ============================================================
# SPARSE CODING (NO DENSE STORAGE)
# ============================================================

def sparse_code(
    mask: np.ndarray,
    D: torch.Tensor,
    device="cuda",
    downsample_factor=2,
    n_iters=400,
    lambda_sparse=0.6,
    threshold_init=0.01,
    threshold_final=0.02,
):
    mask_ds = downsample_mask(mask.astype(np.float32), downsample_factor)
    sdt = compute_sdt(mask_ds)
    X = torch.tensor(sdt, device=device, dtype=torch.float32)[None, None]

    K = D.shape[0]
    pad = D.shape[2] // 2
    Z = nn.Parameter(torch.zeros(1, K, *X.shape[2:], device=device))

    opt = torch.optim.Adam([Z], lr=0.1)

    for it in tqdm(range(n_iters), desc="Sparse coding"):
        thr = threshold_init + (threshold_final - threshold_init) * it / n_iters
        opt.zero_grad()
        X_hat = F.conv3d(Z, D, padding=pad, groups=K).sum(1, keepdim=True)
        loss = 0.5 * F.mse_loss(X_hat, X) + lambda_sparse * Z.abs().mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            Z.copy_(torch.sign(Z) * F.relu(Z.abs() - thr))

    return dense_to_sparse(Z, threshold_final)


# ============================================================
# RECONSTRUCTION FROM SPARSE Z
# ============================================================

def reconstruct_mask_from_sparse(
    idx, vals, shape, D, upsample_factor=2
):
    device = D.device
    Z = torch.zeros((1, *shape), device=device)
    Z[0, idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = vals

    pad = D.shape[2] // 2
    sdf = F.conv3d(Z, D, padding=pad, groups=D.shape[0]).sum(1)
    mask = (sdf[0] >= 0).float()
    return upsample_mask(mask, upsample_factor)


# ============================================================
# MAIN
# ============================================================

data_path = Path("./data/vesuvius-challenge-surface-detection")
save_path = Path("./data/train_cdl_labels")
save_path.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(data_path / "train.csv")
    device = "cuda"

    # ---- train dictionary once ----
    ref_id = df.iloc[0]["id"]
    mask = load_volume(data_path / "train_labels" / f"{ref_id}.tif")
    mask = mask * (mask != 2)

    D = train_dictionary(mask, device=device)
    save_sparse_z(save_path / "dictionary.npz",
                  *dense_to_sparse(D[None], 0.0))

    # ---- sparse code all cases ----
    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_id = row["id"]
        out_file = save_path / f"{case_id}.npz"
        if out_file.exists():
            continue

        mask = load_volume(data_path / "train_labels" / f"{case_id}.tif")
        mask = mask * (mask != 2)

        idx, vals, shape = sparse_code(mask, D, device=device)
        save_sparse_z(out_file, idx, vals, shape)


if __name__ == "__main__":
    main()
