import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import WeightedEulerCurve
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

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

    def __init__(self, eps=0.5, max_iter=100):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.euler_curve = WeightedEulerCurve()
        self.wdist = SinkhornDistance(
            eps=eps,
            max_iter=max_iter,
            reduction=None
        )

    def _compute_summary(self, x):
        # keep raw Euler curves (important!)
        return self.euler_curve(x)

    def forward(self, pred, gt, loss_mask=None):
        B = pred.shape[0]

        if loss_mask is not None:
            pred = pred * loss_mask

        total = 0.0

        for b in range(B):
            s_pred = self._compute_summary(pred[b])
            s_gt   = self._compute_summary(gt[b])

            loss_b, _, _ = self.wdist(s_pred, s_gt)
            total += loss_b

        return total / B



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