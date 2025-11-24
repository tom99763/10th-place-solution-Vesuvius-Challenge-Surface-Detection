import torch
import torch.nn.functional as F

# ---------------------
# Jacobian Determinant
# ---------------------
def jacobian_determinant(flow):
    # flow: (B,3,D,H,W) voxel displacement
    dz = torch.gradient(flow[:, 2], dim=1)[0]  # ∂z/∂z
    dy = torch.gradient(flow[:, 1], dim=2)[0]  # ∂y/∂y
    dx = torch.gradient(flow[:, 0], dim=3)[0]  # ∂x/∂x

    j11 = dx + 1
    j22 = dy + 1
    j33 = dz + 1

    j12 = torch.gradient(flow[:, 0], dim=2)[0]
    j13 = torch.gradient(flow[:, 0], dim=1)[0]
    j21 = torch.gradient(flow[:, 1], dim=3)[0]
    j23 = torch.gradient(flow[:, 1], dim=1)[0]
    j31 = torch.gradient(flow[:, 2], dim=3)[0]
    j32 = torch.gradient(flow[:, 2], dim=2)[0]

    det = (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
    return det