from scipy.ndimage import (distance_transform_edt, binary_dilation,
                           grey_dilation, gaussian_gradient_magnitude, map_coordinates)
import numpy as np
import networkx as nx


class ExactScrollDecomposition:
    """
    Exact scroll mask decomposition using:
    - sdf: signed distance field
    - normals: gradient of sdf
    - thickness: distance to mask boundary along normal
    Fully vectorized inverse transform using sparse accumulation.
    """

    def __init__(self, sdf_truncate=20.0, substeps=5):
        self.sdf_truncate = sdf_truncate
        self.substeps = substeps  # sub-voxel sampling along normal

    def transform(self, mask):
        mask = mask.astype(bool)
        self.mask_shape = mask.shape
        sdf = self._compute_sdf(mask)
        normals = self._normals_from_sdf(sdf)
        thickness = self._compute_thickness(mask)
        return {"sdf": sdf, "normals": normals, "thickness": thickness}

    def inverse_transform(self, pred):
        D, H, W = self.mask_shape
        sdf = pred["sdf"]
        normals = pred["normals"]
        thickness = pred["thickness"]

        # --- Center surface coordinates ---
        center_mask = sdf <= 0
        coords = np.argwhere(center_mask)
        if coords.size == 0:
            return np.zeros((D, H, W), dtype=np.uint8)

        dz = normals[0][center_mask]
        dy = normals[1][center_mask]
        dx = normals[2][center_mask]
        t_half = thickness[center_mask] / 2.0

        # --- Sub-voxel steps along normals ---
        num_centers = coords.shape[0]
        steps = np.linspace(-1, 1, self.substeps)[None, :] * t_half[:, None]

        z_coords = coords[:, 0:1] + dz[:, None] * steps
        y_coords = coords[:, 1:2] + dy[:, None] * steps
        x_coords = coords[:, 2:3] + dx[:, None] * steps

        # --- Round and clip voxel indices ---
        z_idx = np.clip(np.round(z_coords).astype(int), 0, D-1).ravel()
        y_idx = np.clip(np.round(y_coords).astype(int), 0, H-1).ravel()
        x_idx = np.clip(np.round(x_coords).astype(int), 0, W-1).ravel()

        # --- Sparse accumulation ---
        flat_idx = z_idx*H*W + y_idx*W + x_idx
        ones = np.ones_like(flat_idx, dtype=np.uint16)
        recon_sparse = coo_matrix((ones, (flat_idx, np.zeros_like(flat_idx))),
                                  shape=(D*H*W, 1))

        # Convert to dense mask
        recon_flat = (recon_sparse.toarray().ravel() > 0).astype(np.uint8)
        recon = recon_flat.reshape(D, H, W)

        return recon

    def _compute_sdf(self, mask):
        dist_out = distance_transform_edt(~mask)
        dist_in = distance_transform_edt(mask)
        sdf = dist_out - dist_in
        return np.clip(sdf, -self.sdf_truncate, self.sdf_truncate).astype(np.float32)

    def _normals_from_sdf(self, sdf):
        dz, dy, dx = np.gradient(sdf)
        grad = np.stack([dz, dy, dx], axis=0)
        norm = np.linalg.norm(grad, axis=0, keepdims=True) + 1e-8
        return (grad / norm).astype(np.float32)

    def _compute_thickness(self, mask):
        dist_in = distance_transform_edt(mask)
        dist_out = distance_transform_edt(~mask)
        thickness = dist_in + dist_out
        return thickness.astype(np.float32)