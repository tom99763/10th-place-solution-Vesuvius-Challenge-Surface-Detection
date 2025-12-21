import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from scipy.ndimage import distance_transform_edt
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from src.procs.proc_data import *
from cv import calc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ============================================================
# Utilities: downsample / upsample
# ============================================================

def downsample_mask(mask_np, factor):
    """Topology-preserving downsampling via max-pooling."""
    if factor == 1:
        return mask_np
    x = torch.tensor(mask_np, dtype=torch.float32)[None, None]
    x_ds = F.max_pool3d(x, kernel_size=factor, stride=factor)
    return x_ds[0, 0].cpu().numpy()


def upsample_mask(mask_ds, factor):
    """Nearest-neighbor upsampling back to original resolution."""
    if factor == 1:
        return mask_ds
    x = torch.tensor(mask_ds, dtype=torch.float32)[None, None]
    x_up = F.interpolate(x, scale_factor=factor, mode="nearest")
    return x_up[0, 0]


# ============================================================
# CDL with SDT (downsampled) and strong sparsity (~40%)
# ============================================================

def cdl_sdt_sparse(
    binary_mask_np,
    device="cuda",
    K=64,
    kernel_size=9,
    sdt_clip=20.0,
    lambda_sparse=0.8,
    lambda_D=0.01,
    n_epochs=200,
    threshold_init=0.01,
    threshold_final=0.05,
    seed=42,
    downsample_factor=2,
):
    """
    3D Convolutional Dictionary Learning with SDT and ~40% sparsity.
    """

    # -------------------------
    # 0. Determinism
    # -------------------------
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------------
    # 1. Downsample mask
    # -------------------------
    binary_mask_np = binary_mask_np.astype(np.float32)
    binary_mask_ds = downsample_mask(binary_mask_np, downsample_factor)

    # -------------------------
    # 2. SDT
    # -------------------------
    sdt = distance_transform_edt(binary_mask_ds) - distance_transform_edt(1 - binary_mask_ds)
    sdt = np.clip(sdt, -sdt_clip, sdt_clip)
    X = torch.tensor(sdt, dtype=torch.float32, device=device)[None, None]

    # -------------------------
    # 3. Kernel size
    # -------------------------
    kernel_size = max(3, kernel_size // downsample_factor)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    # -------------------------
    # 4. Dictionary D
    # -------------------------
    class ConvDictionary(nn.Module):
        def __init__(self, K, k, seed):
            super().__init__()
            g = torch.Generator(device=device).manual_seed(seed)
            self.D = nn.Parameter(torch.randn(K, 1, k, k, k, generator=g, device=device))
            self.normalize()

        def normalize(self):
            with torch.no_grad():
                self.D /= torch.norm(self.D.flatten(1), dim=1).view(-1,1,1,1,1) + 1e-8

    dict_model = ConvDictionary(K, kernel_size, seed)

    # -------------------------
    # 5. Sparse codes Z
    # -------------------------
    gZ = torch.Generator(device=device).manual_seed(seed)
    Z = nn.Parameter(torch.randn(1, K, *X.shape[2:], generator=gZ, device=device))

    opt_D = torch.optim.Adam([dict_model.D], lr=1e-3)
    opt_Z = torch.optim.Adam([Z], lr=1e-2)

    # -------------------------
    # 6. Training loop
    # -------------------------
    loop = tqdm(range(n_epochs), desc="CDL")
    for epoch in loop:
        # Threshold annealing
        threshold = threshold_init + (threshold_final - threshold_init) * (epoch / n_epochs)

        # ---- Z update ----
        for _ in range(10): #10
            opt_Z.zero_grad()
            X_hat = F.conv3d(Z, dict_model.D, padding=pad, groups=K).sum(1, keepdim=True)
            loss_recon = 0.5 * F.mse_loss(X_hat, X)
            loss_sparse = lambda_sparse * Z.abs().mean()
            (loss_recon + loss_sparse).backward()
            opt_Z.step()

            # Hard soft-thresholding
            with torch.no_grad():
                Z_abs = Z.data.abs()
                Z.data = torch.sign(Z.data) * F.relu(Z_abs - threshold)

        # ---- D update ----
        opt_D.zero_grad()
        X_hat = F.conv3d(Z, dict_model.D, padding=pad, groups=K).sum(1, keepdim=True)
        loss_D = 0.5 * F.mse_loss(X_hat, X) + lambda_D * dict_model.D.abs().mean()
        loss_D.backward()
        opt_D.step()
        dict_model.normalize()

        # Progress
        sparse_rate_Z = (Z.abs() < 1e-8).float().mean().item()
        sparse_rate_D = (dict_model.D.abs() < threshold).float().mean().item()
        loop.set_postfix(loss=f"recon:{loss_recon.item():.4f} sparseZ:{sparse_rate_Z:.3f} sparseD:{sparse_rate_D:.3f}")

    # -------------------------
    # Reconstruction low-res
    # -------------------------
    with torch.no_grad():
        X_hat = F.conv3d(Z, dict_model.D, padding=pad, groups=K).sum(1)
        mask_hat_ds = (X_hat[0] > 0).float()

    # -------------------------
    # Upsample reconstruction
    # -------------------------
    mask_hat = upsample_mask(mask_hat_ds, downsample_factor)

    # Final threshold dictionary
    with torch.no_grad():
        dict_model.D[dict_model.D.abs() < threshold] = 0

    return dict_model.D.detach(), Z.detach(), mask_hat


# ============================================================
# Sparse coding for new masks
# ============================================================

def compute_sparse_code_from_mask_sparse(
    mask_np,
    D,
    n_iters=400,
    lr=0.1,
    lambda_sparse=0.8,
    threshold_init=0.01,
    threshold_final=0.02,
    sdt_clip=20.0,
    device="cuda",
    downsample_factor=2,
):
    mask_ds = downsample_mask(mask_np.astype(np.float32), downsample_factor)

    # SDT
    sdt = distance_transform_edt(mask_ds) - distance_transform_edt(1 - mask_ds)
    sdt = np.clip(sdt, -sdt_clip, sdt_clip)
    X = torch.tensor(sdt, dtype=torch.float32, device=device)[None, None]

    K = D.shape[0]
    pad = D.shape[2] // 2
    Z = torch.zeros(1, K, *X.shape[2:], device=device, requires_grad=True)

    opt = torch.optim.Adam([Z], lr=lr)
    loop = tqdm(range(n_iters), desc="Sparse coding")
    for epoch in loop:
        # Threshold annealing
        threshold = threshold_init + (threshold_final - threshold_init) * (epoch / n_iters)

        opt.zero_grad()
        X_hat = F.conv3d(Z, D, padding=pad, groups=K).sum(1, keepdim=True)
        loss_recon = F.mse_loss(X_hat, X)
        loss_sparse = lambda_sparse * Z.abs().mean()
        loss = 0.5 * loss_recon + loss_sparse
        loss.backward()
        opt.step()

        # Hard soft-thresholding
        with torch.no_grad():
            Z_abs = Z.data.abs()
            Z.data = torch.sign(Z.data) * F.relu(Z_abs - threshold)

        loop.set_postfix(loss=f"recon:{loss_recon.item():.4f} sparse:{(Z.abs()<threshold).float().mean().item():.3f}")

    return Z.detach()



def reconstruct_mask(Z, D, upsample_factor=2):
    """
    Reconstruct a binary mask from sparse codes Z and dictionary D,
    and optionally upsample it to the original resolution.

    Args:
        Z (torch.Tensor): Sparse codes, shape (1, K, D, H, W)
        D (torch.Tensor): Dictionary, shape (K, 1, k, k, k)
        upsample_factor (int): Factor to upsample the reconstructed mask

    Returns:
        torch.Tensor: Reconstructed mask, float tensor 0/1 at full resolution
    """
    K = D.shape[0]
    pad = D.shape[2] // 2
    Z = Z.to(D.device)

    # Reconstruct SDT
    sdf_hat = F.conv3d(Z, D, padding=pad, groups=K).sum(dim=1, keepdim=True)

    # Threshold to binary mask
    mask_hat = (sdf_hat >= 0).float()[0, 0]

    # Upsample to original resolution if needed
    if upsample_factor > 1:
        mask_hat = upsample_mask(mask_hat, upsample_factor)

    return mask_hat


# ============================================================
# I/O helpers
# ============================================================

def save_sparse_tensor(tensor: np.ndarray, path: str):
    idx = np.nonzero(tensor)
    values = tensor[idx]
    np.savez_compressed(path, idx=idx, values=values, shape=tensor.shape)


# ============================================================
# Main
# ============================================================

data_path = Path("./data/vesuvius-challenge-surface-detection")
save_path = Path("./data/train_cdl_labels")
save_path.mkdir(exist_ok=True, parents=True)


def main():
    df = pd.read_csv(data_path / "train.csv")

    chosen_id = "1006462223"
    mask = load_volume(data_path / "train_labels" / f"{chosen_id}.tif")
    mask = mask * (mask != 2)

    # ---- Train dictionary ----
    D, _, _ = cdl_sdt_sparse(
        mask,
        downsample_factor=2,
        lambda_sparse=0.6,
        lambda_D=0.01,
        n_epochs=200,
        threshold_init=0.01,
        threshold_final=0.02,
    )
    D[D.abs()<1e-8] = 0.
    sparse_rate_D = ((D.abs() < 1e-8).sum() / D.numel()).item()
    print(f"dictionary sparse rate: {sparse_rate_D:.3f}")
    save_sparse_tensor(D.cpu().numpy(), save_path / "dictionary_sparse.npz")

    # ---- Sparse codes ----
    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_id = row["id"]
        out_file = save_path / f"{case_id}.npz"
        if out_file.exists():
            continue

        mask = load_volume(data_path / "train_labels" / f"{case_id}.tif")
        mask = mask * (mask != 2)

        Z = compute_sparse_code_from_mask_sparse(
            mask,
            D,
            n_iters=600,
            lambda_sparse=0.6,
            threshold_init=0.01,
            threshold_final=0.02,
            downsample_factor=2,
        ).cpu().numpy()[0]
        Z[np.abs(Z)<1e-8] = 0
        sparse_rate_Z = (np.abs(Z) < 1e-8).sum() / Z.size
        print(f"{case_id} sparse rate: {sparse_rate_Z:.3f}")
        mask_hat = reconstruct_mask(torch.tensor(Z[None], device=D.device), D)
        #print(mask.shape, mask_hat.shape)
        # score_report = calc_score(mask, mask_hat.cpu().numpy())
        # print(
        #     "total_score:", score_report.score,
        #     "topo_score:", score_report.topo.toposcore,
        #     "voi_score:", score_report.voi.voi_score,
        #     "surface_dice:", score_report.surface_dice)
        save_sparse_tensor(Z, out_file)


if __name__ == "__main__":
    main()