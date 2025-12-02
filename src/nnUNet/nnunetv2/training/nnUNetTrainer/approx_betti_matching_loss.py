import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class GPUWeightedEulerCurve(nn.Module):
    def __init__(self, num_steps=64, sigma=1e-2, device=None):
        super().__init__()
        self.num_steps = int(num_steps)
        self.sigma = float(sigma)
        self.device = device

    def forward(self, x):
        """
        Robust input normalization for shapes:
         - (H,W) -> single sample 2D
         - (C,H,W) with C==1 or C==3 -> single sample channels collapsed
         - (D,H,W) -> single sample 3D
         - (B,H,W) -> batched single-channel
         - (B,C,H,W) with C>1 -> batched channels collapsed
         - (B,C,D,H,W) -> batched volumetric (collapses channel dim if C>1)
        Returns: (B, T) or (T,) for single-sample inputs
        """
        single_sample = False

        # ---- Normalize to (B, 1, H, W) or (B, 1, D, H, W) ----
        if x.dim() == 2:
            # (H, W) -> (1,1,H,W)
            x = x.unsqueeze(0).unsqueeze(0)
            single_sample = True

        elif x.dim() == 3:
            # ambiguous: could be (C,H,W) or (D,H,W)
            C_or_D, H, W = x.shape
            # Heuristic: treat common channel counts as channels (1 or 3)
            if C_or_D in (1, 3):
                # (C,H,W) -> (1,C,H,W) then collapse channels to single scalar
                x = x.unsqueeze(0)  # (1, C, H, W)
                x = x.mean(dim=1, keepdim=True)  # (1, 1, H, W)
                single_sample = True
            else:
                # treat as (D,H,W) depth volume -> (1,1,D,H,W)
                x = x.unsqueeze(0).unsqueeze(0)
                single_sample = True

        elif x.dim() == 4:
            # could be (B,H,W) or (B,C,H,W) or (C,D,H,W) (rare)
            B, a, b, c = x.shape
            if a > 1 and a <= 4:
                # assume (B, C, H, W) with small channel dimension (1 or 3)
                # collapse channels
                x = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
            else:
                # assume (B, H, W) -> add channel dim
                x = x.unsqueeze(1)  # (B,1,H,W)

        elif x.dim() == 5:
            # (B, C, D, H, W) or (B,1,D,H,W)
            B, C, D, H, W = x.shape
            if C > 1:
                # collapse channels
                x = x.mean(dim=1, keepdim=True)  # (B,1,D,H,W)
            # else keep as is (B,1,D,H,W)

        else:
            raise ValueError(f"Unsupported tensor shape {x.shape}")

        # Now x is either (B,1,H,W) or (B,1,D,H,W)
        is_3d = (x.dim() == 5)
        device = x.device
        B = x.shape[0]

        # thresholds (global across batch for stability)
        x_min = x.amin(dim=tuple(range(1, x.dim())), keepdim=True)
        x_max = x.amax(dim=tuple(range(1, x.dim())), keepdim=True)
        global_min = float(x_min.min().item())
        global_max = float(x_max.max().item())
        if global_max == global_min:
            global_max = global_min + 1e-6

        T = self.num_steps
        thresholds = torch.linspace(global_min, global_max, steps=T, device=device)

        # compute soft fields: expand thresholds and broadcast
        # collapse channel dim (we already ensured channel dim == 1)
        x_spatial = x.squeeze(1)  # (B, H, W) or (B, D, H, W)
        if is_3d:
            # x_spatial: (B, D, H, W)
            t = thresholds.view(1, T, 1, 1, 1)  # (1,T,1,1,1)
            soft = torch.sigmoid((x_spatial.unsqueeze(1) - t) / (self.sigma + 1e-12))
            # soft: (B, T, D, H, W)
        else:
            # x_spatial: (B, H, W)
            t = thresholds.view(1, T, 1, 1)  # (1,T,1,1)
            soft = torch.sigmoid((x_spatial.unsqueeze(1) - t) / (self.sigma + 1e-12))
            # soft: (B, T, H, W)

        # --- Euler computation (unchanged from your implementation) ---
        if is_3d:
            B, T, D, H, W = soft.shape
            V = soft.view(B, T, -1).sum(dim=-1)

            def shift_dim(tensor, dim, offset=1):
                # pad for last dims: (W_left, W_right, H_left, H_right, D_left, D_right)
                if dim == 4:  # W
                    return F.pad(tensor, (0, offset, 0, 0, 0, 0))[:, :, :, :, :-offset]
                if dim == 3:  # H
                    return F.pad(tensor, (0, 0, 0, offset, 0, 0))[:, :, :, :-offset, :]
                if dim == 2:  # D
                    return F.pad(tensor, (0, 0, 0, 0, 0, offset))[:, :, :-offset, :, :]

            s = soft
            s_d = shift_dim(s, 2, 1)
            s_h = shift_dim(s, 3, 1)
            s_w = shift_dim(s, 4, 1)

            E_d = (s * s_d).view(B, T, -1).sum(dim=-1)
            E_h = (s * s_h).view(B, T, -1).sum(dim=-1)
            E_w = (s * s_w).view(B, T, -1).sum(dim=-1)
            E = E_d + E_h + E_w

            s_hw = s * shift_dim(s, 4, 1) * shift_dim(s, 3, 1) * shift_dim(shift_dim(s, 3, 1), 4, 1)
            F_hw = s_hw.view(B, T, -1).sum(dim=-1)
            s_dh = s * shift_dim(s, 3, 1) * shift_dim(s, 2, 1) * shift_dim(shift_dim(s, 2, 1), 3, 1)
            F_dh = s_dh.view(B, T, -1).sum(dim=-1)
            s_dw = s * shift_dim(s, 4, 1) * shift_dim(s, 2, 1) * shift_dim(shift_dim(s, 2, 1), 4, 1)
            F_dw = s_dw.view(B, T, -1).sum(dim=-1)
            F_total = F_hw + F_dh + F_dw

            s_c = s * shift_dim(s, 4, 1) * shift_dim(s, 3, 1) * shift_dim(shift_dim(s, 3, 1), 4, 1) \
                  * shift_dim(s, 2, 1) * shift_dim(shift_dim(s, 2, 1), 4, 1) * shift_dim(shift_dim(s, 2, 1), 3, 1) \
                  * shift_dim(shift_dim(shift_dim(s, 2, 1), 3, 1), 4, 1)
            C = s_c.view(B, T, -1).sum(dim=-1)
            euler = V - E + F_total - C

        else:
            B, T, H, W = soft.shape
            V = soft.view(B, T, -1).sum(dim=-1)

            def shift2d(tensor, dim, offset=1):
                if dim == 3:  # W
                    return F.pad(tensor, (0, offset, 0, 0))[:, :, :, :-offset]
                if dim == 2:  # H
                    return F.pad(tensor, (0, 0, 0, offset))[:, :, :-offset, :]

            s = soft
            s_h = shift2d(s, 2, 1)
            s_w = shift2d(s, 3, 1)
            E_h = (s * s_h).view(B, T, -1).sum(dim=-1)
            E_w = (s * s_w).view(B, T, -1).sum(dim=-1)
            E = E_h + E_w

            s_hw = s * shift2d(s, 3, 1) * shift2d(s, 2, 1) * shift2d(shift2d(s, 2, 1), 3, 1)
            F_total = s_hw.view(B, T, -1).sum(dim=-1)

            euler = V - E + F_total

        # return shape (T,) for single sample else (B, T)
        if single_sample:
            return euler.squeeze(0)  # (T,)
        return euler  # (B, T)


class SinkhornDistance(nn.Module):

    def __init__(self, eps=0.5, max_iter=100, reduction='none'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        C = self._cost_matrix(x, y)

        batch = x.shape[0] if x.dim() == 3 else 1
        n, m = x.shape[-2], y.shape[-2]

        mu = torch.full((batch, n), 1.0 / n, device=x.device)
        nu = torch.full((batch, m), 1.0 / m, device=y.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        for _ in range(self.max_iter):
            u_prev = u.clone()
            u = self.eps * (torch.log(mu + 1e-8) -
                            torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) -
                            torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v

            if (u - u_prev).abs().mean() < 1e-2:
                break

        pi = torch.exp(self.M(C, u, v))

        # ======================================================
        #  Balanced normalization:
        #    - do NOT divide by n*m (keeps scale meaningful)
        #    - only divide by (max(n, m))
        #      → reduces magnitude but preserves relative strength
        # ======================================================
        raw_cost = torch.sum(pi * C, dim=(-2, -1))
        cost = raw_cost / max(n, m)

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        return torch.sum((x.unsqueeze(-2) - y.unsqueeze(-3)).abs() ** p, dim=-1)


class ApproxBettiMatchingLoss(nn.Module):

    def __init__(self, eps=0.1, max_iter=50, euler_steps=32, euler_sigma=5e-3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.euler_curve = GPUWeightedEulerCurve(num_steps=euler_steps, sigma=euler_sigma, device=self.device)
        self.wdist = SinkhornDistance(
            eps=eps,
            max_iter=max_iter,
            reduction=None
        )

    def _compute_summary(self, x):
        s = self.euler_curve(x)  # (T,)
        # Normalize curve to zero-mean, unit-variance and bounded magnitude
        s = (s - s.mean()) / (s.std() + 1e-6)
        return s

    def forward(self, pred, gt, loss_mask=None):
        B = pred.shape[0]
        pred = pred.softmax(dim=1)
        if loss_mask is not None:
            pred = pred * loss_mask

        # ignore background
        pred = pred[:, 1]
        gt = gt[:, 0]

        total = 0.0
        for b in range(B):
            s_pred = self._compute_summary(pred[b])
            s_gt = self._compute_summary(gt[b])

            # Ensure shapes (T,) -> (1, T) for Sinkhorn (it expects last two dims as positions)
            # SinkhornDistance._cost_matrix expects x,y of shape (B, n, d) or similar.
            # Here, the "points" are 1D positions along threshold axis: we can treat them as
            # 1D coordinates with value = threshold index and weight = euler value.
            # However we already calculate curves; to reuse existing wdist (which expects point clouds),
            # we'll use a simple representation: treat curve as sequence of scalar "features" at positions.
            # Convert each curve into positions along a line: coords = thresholds (0..T-1) and features = curve
            # For simplicity, create 2D points (pos, value) so the cost matrix is meaningful.

            # Make coordinates
            t_len = s_pred.shape[-1] if s_pred.dim() > 0 else s_pred.shape[0]
            coords = torch.arange(t_len, device=s_pred.device, dtype=s_pred.dtype)
            coords = coords.view(-1, 1)  # (T,1)

            # create point clouds: (1, T, 2) -> (pos, value)
            pc_pred = torch.stack(
                [coords.squeeze(-1), s_pred.detach().cpu() if s_pred.device.type == 'cpu' else s_pred],
                dim=1).unsqueeze(0).to(s_pred.device)
            pc_gt = torch.stack([coords.squeeze(-1), s_gt.detach().cpu() if s_gt.device.type == 'cpu' else s_gt],
                                dim=1).unsqueeze(0).to(s_gt.device)

            # call Sinkhorn (it handles batches). returns (batch_costs, pi, C)
            loss_b, _, _ = self.wdist(pc_pred, pc_gt)
            total += loss_b

        return 100 * total / B



class BettiDicCELosss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        betti_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        weight_betti=1,
        ignore_label = None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_betti = weight_betti
        self.ignore_label = ignore_label

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