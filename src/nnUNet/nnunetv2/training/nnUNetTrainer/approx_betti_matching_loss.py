import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import WeightedEulerCurve
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

# ===========================================================
# ============== Normalized Sinkhorn Distance ===============
# ===========================================================
class SinkhornDistance(nn.Module):
    """
    Entropy-regularized OT distance with:
    - normalized cost
    - per-point average cost instead of sum
    - true Wasserstein-2 distance (sqrt)
    """

    def __init__(self, eps=1.0, max_iter=100, reduction='none'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # Cost matrix |x_i - y_j|^2 (but small due to normalization)
        C = self._cost_matrix(x, y)

        batch = x.shape[0] if x.dim() == 3 else 1
        n, m = x.shape[-2], y.shape[-2]

        mu = torch.full((batch, n), 1.0 / n, device=x.device)
        nu = torch.full((batch, m), 1.0 / m, device=y.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        for _ in range(self.max_iter):
            u_prev = u.clone()
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            if (u - u_prev).abs().mean() < 1e-2:
                break

        pi = torch.exp(self.M(C, u, v))

        # -------- Normalized OT cost --------
        cost = torch.sum(pi * C, dim=(-2, -1))          # sum over all pairs
        cost = cost / (n * m)                           # normalize per point
        cost = torch.sqrt(cost + 1e-8)                  # true Wasserstein-2

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        return torch.sum((torch.abs(x_col - y_lin)) ** p, dim=-1)


# ===========================================================
# ========== Approximate Betti Matching Loss ================
# ===========================================================
class ApproxBettiMatchingLoss(nn.Module):
    """
    Uses:
    - normalized Euler summaries
    - normalized Sinkhorn distance
    """

    def __init__(self, eps=1.0, max_iter=100):
        super().__init__()
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.euler_curve = WeightedEulerCurve()
        self.wdist = SinkhornDistance(eps=eps, max_iter=max_iter, reduction=None)

    def _compute_summary(self, x):
        summary = self.euler_curve(x)
        # ---- normalize Euler summaries to unit sphere ----
        summary = summary / (summary.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return summary

    def forward(self, pred, gt, loss_mask=None):
        B = pred.shape[0]
        if loss_mask is not None:
            pred = pred * loss_mask
        total_loss = 0.0

        for b in range(B):
            sp = self._compute_summary(pred[b])
            sg = self._compute_summary(gt[b])
            loss_b, _, _ = self.wdist(sp, sg)
            total_loss += loss_b

        return total_loss / B



class BettiDicCELosss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        betti_kwargs,
        ce_kwargs,
        weight_ce=0.1,
        weight_dice=1,
        weight_betti=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

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