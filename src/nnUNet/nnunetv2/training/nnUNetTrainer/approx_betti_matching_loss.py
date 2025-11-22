import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import WeightedEulerCurve
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from geomloss import SamplesLoss

class EulerCurve3D(nn.Module):
    """
    Fixed 3D Euler Characteristic Curve. Now handles dims correctly!
    Input: (B, 1, H, W, D) in [0,1]
    Output: (B, N, 2) → (t, χ(t))
    """
    def __init__(self, resolution: int = 400, sharpness: int = 100):
        super().__init__()
        self.resolution = resolution
        self.sharpness = sharpness
        self.register_buffer('thresholds', torch.linspace(0, 1, resolution + 1)[:-1])

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D_ = vol.shape  # D_ to avoid conflict with depth dim
        assert C == 1

        curves = torch.zeros(B, self.resolution, 2, device=vol.device)
        curves[:, :, 0] = self.thresholds.unsqueeze(0)  # Filtration values

        # Vectorized binary masks over thresholds: (B, N, H, W, D_)
        vol_flat = vol.view(B, 1, -1)  # (B, 1, H*W*D_)
        thresh_flat = self.thresholds.view(1, -1, 1)  # (1, N, 1)
        binary_flat = torch.sigmoid(self.sharpness * (vol_flat - thresh_flat))  # (B, N, Voxels)

        # But for conv, we need spatial dims → reshape back per N
        # Simple loop over N: fast since N=400, each conv is O(HWD)
        for n in range(self.resolution):
            binary_n = binary_flat[:, n].view(B, 1, H, W, D_)  # (B, 1, H, W, D_) per threshold

            # Vertices: sum of voxels
            V = binary_n.sum(dim=(2, 3, 4))  # (B,)

            # Edges: conv3d with proper kernels and padding
            # X-edges (along depth): kernel (2,1,1), pad depth=1
            kernel_x = torch.ones(1, 1, 2, 1, 1, device=vol.device) / 2
            E_x = F.conv3d(binary_n, kernel_x, padding=(1, 0, 0)).sum(dim=(2, 3, 4))  # (B,)

            # Y-edges (along height): kernel (1,2,1), pad height=1
            kernel_y = torch.ones(1, 1, 1, 2, 1, device=vol.device) / 2
            E_y = F.conv3d(binary_n, kernel_y, padding=(0, 1, 0)).sum(dim=(2, 3, 4))

            # Z-edges (along width): kernel (1,1,2), pad width=1
            kernel_z = torch.ones(1, 1, 1, 1, 2, device=vol.device) / 2
            E_z = F.conv3d(binary_n, kernel_z, padding=(0, 0, 1)).sum(dim=(2, 3, 4))
            E = E_x + E_y + E_z  # (B,)

            # Faces: 2D faces in 3 orientations
            # XY-faces (2x2x1), pad (1,1,0)
            k_xy = torch.ones(1, 1, 2, 2, 1, device=vol.device) / 4
            F_xy = F.conv3d(binary_n, k_xy, padding=(1, 1, 0)).sum(dim=(2, 3, 4))

            # XZ-faces (2x1x2), pad (1,0,1)
            k_xz = torch.ones(1, 1, 2, 1, 2, device=vol.device) / 4
            F_xz = F.conv3d(binary_n, k_xz, padding=(1, 0, 1)).sum(dim=(2, 3, 4))

            # YZ-faces (1x2x2), pad (0,1,1)
            k_yz = torch.ones(1, 1, 1, 2, 2, device=vol.device) / 4
            F_yz = F.conv3d(binary_n, k_yz, padding=(0, 1, 1)).sum(dim=(2, 3, 4))
            F_total = F_xy + F_xz + F_yz  # (B,)

            # Cubes: 2x2x2, pad (1,1,1)
            k_cube = torch.ones(1, 1, 2, 2, 2, device=vol.device) / 8
            C = F.conv3d(binary_n, k_cube, padding=(1, 1, 1)).sum(dim=(2, 3, 4))  # (B,)

            # Euler: V - E + F - C
            chi = V - E + F_total - C
            curves[:, n, 1] = chi.float()

        return curves


class ApproxBettiMatchingLoss(nn.Module):
    def __init__(
        self,
        eps: float = 0.05,
        resolution: int = 400,
        target_class: int = 1,
        topo_weight: float = 20.0,
        sharpness: int = 100,  # Threshold sharpness (higher = harder binary)
    ):
        super().__init__()
        self.target_class = target_class
        self.topo_weight = topo_weight
        self.sharpness = sharpness

        self.euler_3d = EulerCurve3D(resolution=resolution, sharpness=sharpness)

        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=eps,
            scaling=0.9,
            backend="tensorized",
        )

    def _volume_to_points(self, vol: torch.Tensor) -> torch.Tensor:
        vol = vol.clamp(0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        curve = self.euler_3d(vol)  # (1, N, 2)
        points = curve.squeeze(0)   # (N, 2)

        # Normalize χ to [ -1, 1 ] range (prevents scale imbalance with t ∈ [0,1])
        chi = points[:, 1]
        max_abs_chi = chi.abs().max().clamp(min=1e-8)
        points[:, 1] = chi / max_abs_chi

        return points

    def forward(self, pred_logits: torch.Tensor, gt_labels: torch.Tensor):
        B = pred_logits.shape[0]
        device = pred_logits.device
        pred_prob = torch.softmax(pred_logits, dim=1)
        total_loss = 0.0

        for b in range(B):
            pred_cls = pred_prob[b, self.target_class]

            gt_mask = (gt_labels[b, 0] == self.target_class)
            gt_cls = gt_mask.float()

            if gt_cls.sum() < 100:  # Skip tiny/empty masks
                continue

            pred_pts = self._volume_to_points(pred_cls)
            gt_pts = self._volume_to_points(gt_cls)

            pred_pts = pred_pts.unsqueeze(0).to(device)
            gt_pts = gt_pts.unsqueeze(0).to(device)

            loss_b = self.sinkhorn(pred_pts, gt_pts)
            total_loss += loss_b

        return self.topo_weight * total_loss / max(B, 1)



class BettiDicCELosss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        betti_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        weight_betti=1,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_betti = weight_betti

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.betti = ApproxBettiMatchingLoss(**betti_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables " "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_betti != 0 else 0
        betti_loss = self.betti(net_output, target_dice, loss_mask=mask) if self.weight_betti != 0 else 0
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * betti_loss
        return result