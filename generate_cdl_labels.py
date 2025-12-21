import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from scipy.ndimage import distance_transform_edt
import os
from tqdm import tqdm
from src.procs.proc_data import *
import pandas as pd
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def cdl_sdt_with_betti1_ripser_deterministic(
    binary_mask_np,
    device="cuda",
    K=64,
    kernel_size=9,
    sdt_clip=20.0,
    lambda_sparse=0.01,
    n_epochs=200,
    seed=42,
):
    """
    Fully deterministic 3D Convolutional Dictionary Learning on SDT
    with β1 persistent homology loss (CubicalRipser).

    Args:
        binary_mask_np (np.ndarray): input mask (D,H,W) with 0/1
        device (str): "cuda" or "cpu"
        K (int): number of convolutional atoms
        kernel_size (int): size of 3D convolutional atoms
        sdt_clip (float): clip SDT values
        lambda_sparse (float): weight for L1 sparsity
        lambda_betti1 (float): weight for PH β1 loss
        n_epochs (int): number of training epochs
        seed (int): deterministic seed

    Returns:
        dict_atoms (torch.Tensor): (K,1,k,k,k)
        sparse_codes (torch.Tensor): (1,K,D,H,W)
        reconstructed_mask (torch.Tensor): (D,H,W) binary
    """

    binary_mask_np = binary_mask_np.astype('float32')

    # -------------------------
    # 0. Set deterministic seeds
    # -------------------------
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------------
    # 1. SDT preprocessing
    # -------------------------
    sdt = distance_transform_edt(binary_mask_np) - distance_transform_edt(1 - binary_mask_np)
    sdt = np.clip(sdt, -sdt_clip, sdt_clip)
    X = torch.tensor(sdt, dtype=torch.float32, device=device)[None, None]  # (1,1,D,H,W)

    # -------------------------
    # 2. CDL dictionary & sparse codes
    # -------------------------
    class ConvDictionary(nn.Module):
        def __init__(self, K, k, seed):
            super().__init__()
            rng = torch.Generator(device=device).manual_seed(seed)
            self.D = nn.Parameter(torch.randn(K, 1, k, k, k, device=device, generator=rng))
            self.normalize()

        def normalize(self):
            with torch.no_grad():
                self.D /= (torch.norm(self.D.flatten(1), dim=1).view(-1,1,1,1,1) + 1e-8)

    dict_model = ConvDictionary(K, kernel_size, seed)

    # deterministic sparse code initialization
    rng_z = torch.Generator(device=device).manual_seed(seed)
    Z = nn.Parameter(torch.randn(1, K, *X.shape[2:], device=device, generator=rng_z))

    opt_D = torch.optim.Adam([dict_model.D], lr=1e-3)
    opt_Z = torch.optim.Adam([Z], lr=1e-2)

    pad = kernel_size // 2

    # -------------------------
    # 3. Training loop
    # -------------------------
    for epoch in tqdm(range(n_epochs)):

        # --- sparse coding update ---
        for _ in range(5):
            opt_Z.zero_grad()

            # grouped conv: Z shape (1,K,D,H,W), D shape (K,1,k,k,k)
            X_hat = F.conv3d(Z, dict_model.D, padding=pad, groups=K)
            X_hat_sum = X_hat.sum(dim=1, keepdim=True)  # (1,1,D,H,W)

            loss_recon = 0.5 * ((X_hat_sum - X) ** 2).mean()
            loss_sparse = lambda_sparse * Z.abs().mean()
            loss = loss_recon + loss_sparse
            loss.backward()
            opt_Z.step()

        # --- dictionary update ---
        opt_D.zero_grad()
        X_hat = F.conv3d(Z, dict_model.D, padding=pad, groups=K)
        X_hat_sum = X_hat.sum(dim=1, keepdim=True)
        loss_D = 0.5 * ((X_hat_sum - X) ** 2).mean()
        loss_D.backward()
        opt_D.step()
        dict_model.normalize()

        if epoch % 20 == 0:
            print(f"[{epoch:03d}] Recon={loss_recon.item():.4f}, Sparse={loss_sparse.item():.4f}")

    # -------------------------
    # 4. Reconstruction & threshold
    # -------------------------
    with torch.no_grad():
        X_hat = F.conv3d(Z, dict_model.D, padding=pad, groups=K)
        X_hat_sum = X_hat.sum(dim=1, keepdim=True)
        mask_hat = (X_hat_sum[0,0] > 0).float()

    return dict_model.D.detach(), Z.detach(), mask_hat


def compute_sparse_code_from_mask(mask_np, D, n_iters=100, lr=0.1,
                                  lambda_sparse=0.01, sdt_clip=20.0, device="cuda"):
    """
    Compute sparse code Z for a new input mask using fixed dictionary D.

    Args:
        mask_np (np.ndarray): input binary mask (D,H,W)
        D (torch.Tensor): fixed dictionary (K,1,k,k,k)
        n_iters (int): number of optimization steps
        lr (float): learning rate for Z optimization
        lambda_sparse (float): weight for sparsity regularization
        sdt_clip (float): clip SDT values
        device (str): "cuda" or "cpu"

    Returns:
        Z (torch.Tensor): optimized sparse code (1,K,D,H,W)
    """
    # 1. Compute SDT
    sdt = distance_transform_edt(mask_np) - distance_transform_edt(1 - mask_np)
    sdt = np.clip(sdt, -sdt_clip, sdt_clip)
    X = torch.tensor(sdt, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    # 2. Initialize Z
    K = D.shape[0]
    pad = D.shape[2] // 2  # kernel_size//2
    Z = torch.zeros(1, K, *X.shape[2:], device=device, requires_grad=True)

    optimizer = torch.optim.Adam([Z], lr=lr)

    # 3. Optimize Z
    for i in tqdm(range(n_iters), desc="Sparse coding"):
        optimizer.zero_grad()
        X_hat = F.conv3d(Z, D, padding=pad, groups=K).sum(dim=1, keepdim=True)
        loss = 0.5 * F.mse_loss(X_hat, X) + lambda_sparse * Z.abs().mean()
        loss.backward()
        optimizer.step()

    return Z.detach()

def reconstruct_mask(Z, D):
    K = D.shape[0]
    pad = D.shape[2] // 2
    sdf_hat = F.conv3d(Z, D, padding=pad, groups=K).sum(dim=1, keepdim=True)
    mask_hat = sdf_hat>=0
    return mask_hat

data_path = Path('./data/vesuvius-challenge-surface-detection')
save_path = Path('./data/train_cdl_labels')

def save_sparse_tensor(tensor: np.ndarray, path: str):
    """
    Save a dense tensor as sparse format: indices, values, shape
    tensor: np.ndarray (any shape)
    path: file path to save .npz
    """
    idx = np.nonzero(tensor)
    values = tensor[idx]
    np.savez_compressed(path, idx=idx, values=values, shape=tensor.shape)

# -----------------------------
# Example usage in main()
# -----------------------------
def main():
    df = pd.read_csv(data_path / 'train.csv')
    chosen_id = "1006462223"
    chosen_path = data_path/'train_labels'/chosen_id
    chosen_mask = load_volume(chosen_path)
    chosen_mask = chosen_mask * (chosen_mask != 2)

    # 1. CDL dictionary
    D, _, _ = cdl_sdt_with_betti1_ripser_deterministic(chosen_mask)
    D[D<1e-8] = 0.
    D_np = D.cpu().numpy()
    save_sparse_tensor(D_np, str(save_path / "dictionary_sparse.npz"))

    # 2. Sparse codes Z for each mask
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generate Sparse Codes"):
        case_id = row['id']
        out_file = save_path / f"{case_id}.npz"
        if out_file.exists():
            continue
        mask_path = data_path / 'train_labels' / f'{case_id}.tif'
        mask = load_volume(mask_path)
        mask = mask * (mask != 2)
        Z = compute_sparse_code_from_mask(mask, D).cpu().numpy()[0]  # shape (K,D,H,W)
        Z[Z<1e-8] = 0.
        save_sparse_tensor(Z, str(out_file))

if __name__ == "__main__":
    main()