"""
Focal Tversky++ Lossを用いるRSNA2025用トレーナー

参考:
- ベース: nnUNet/nnunetv2/training/nnUNetTrainer/project_specific/rsna2025/tversky_loss.py
  - MemoryEfficientTverskyLossに準拠した実装パターン（非線形、バッチ集約、DDP、背景制御、loss_mask）
"""

from typing import Callable

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv3 import (
    RSNA2025Trainer_moreDAv3,
)
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv5 import (
    RSNA2025Trainer_moreDAv5,
)
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class MemoryEfficientFocalTverskyPlusPlusLoss(nn.Module):
    """
    Focal Tversky++ Loss（メモリ効率版）

    Tversky++の定義に基づき、以下を計算します:
        tp = sum(p * g)
        fp = alpha * sum((p * (1 - g)) ** gamma_pp)
        fn = beta  * sum(((1 - p) * g) ** gamma_pp)
        loss = (1 - (tp + smooth_nr) / (tp + fp + fn + smooth_dr)) ** gamma_focal

    Parameters:
        alpha: 偽陽性の重み
        beta: 偽陰性の重み
        gamma_pp: Tversky++の冪指数
        gamma_focal: Focal化の冪指数
        smooth_nr/dr: ゼロ割防止の平滑化項
        apply_nonlin: softmax または sigmoid を想定
        batch_dice: バッチ全体で集約するか
        do_bg: 背景チャネルを含めるか
        ddp: DDP集約を行うか
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma_pp: float = 2.0,
        gamma_focal: float = 4.0 / 3.0,
        apply_nonlin: Callable | None = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        ddp: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_pp = gamma_pp
        self.gamma_focal = gamma_focal
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.ddp = ddp

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y.to(dtype=x.dtype)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:]

        p0 = x
        p1 = 1.0 - p0
        g0 = y_onehot
        g1 = 1.0 - g0

        # 基本項（冪乗の対象）を先に定義
        base_tp = p0 * g0
        base_fp = p0 * g1
        base_fn = p1 * g0

        if loss_mask is None:
            tp = base_tp.sum(axes)
            fp = self.alpha * (base_fp**self.gamma_pp).sum(axes)
            fn = self.beta * (base_fn**self.gamma_pp).sum(axes)
        else:
            # マスクは冪乗の“外側”で適用し、重み付けを線形に保つ
            tp = (base_tp * loss_mask).sum(axes)
            fp = self.alpha * ((base_fp**self.gamma_pp) * loss_mask).sum(axes)
            fn = self.beta * ((base_fn**self.gamma_pp) * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)
            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

            nom = tp + self.smooth_nr
            denom = torch.clip(tp + fp + fn + self.smooth_dr, min=1e-8)
            per_class = (1.0 - nom / denom) ** self.gamma_focal
            loss = per_class.mean()
        else:
            nom = tp + self.smooth_nr
            denom = torch.clip(tp + fp + fn + self.smooth_dr, min=1e-8)
            per_sample_per_class = (1.0 - nom / denom) ** self.gamma_focal
            loss = per_sample_per_class.mean()

        return loss


class RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlus(RSNA2025Trainer_moreDAv3):
    """
    FocalTverskyPlusPlusLossを使用するRSNA2025トレーナー。

    - regions（マルチラベル）の場合は `sigmoid` を用いる。
    - multi-class の場合は `softmax` と `to_onehot_y=True` を用い、背景は除外する。
    - `batch_dice` 設定は FocalTversky++ の `batch` に反映。
    - 係数はTversky同様に FNをやや重めに: alpha=0.3, beta=0.7。
    """

    def _build_loss(self):
        alpha = 0.3
        beta = 0.7
        gamma_pp = 2.0
        gamma_focal = 4.0 / 3.0  # 1.33...

        if self.label_manager.has_regions:
            # マルチラベル（領域）: sigmoid活性 + 背景込み
            loss = MemoryEfficientFocalTverskyPlusPlusLoss(
                alpha=alpha,
                beta=beta,
                gamma_pp=gamma_pp,
                gamma_focal=gamma_focal,
                apply_nonlin=torch.sigmoid,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                ddp=self.is_ddp,
            )
        else:
            # マルチクラス: softmax活性 + 背景除外
            loss = MemoryEfficientFocalTverskyPlusPlusLoss(
                alpha=alpha,
                beta=beta,
                gamma_pp=gamma_pp,
                gamma_focal=gamma_focal,
                apply_nonlin=softmax_helper_dim1,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=False,
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                ddp=self.is_ddp,
            )

        # Deep Supervisionに対応
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # 解像度が低いほど重みを小さく（指数減衰）
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                # DDP時の特殊ケース: 最終出力に微小重み
                weights[-1] = 1e-6
            else:
                # 通常は最終出力を無視
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusWithBackground(RSNA2025Trainer_moreDAv3):
    """
    FocalTverskyPlusPlusLossを使用するRSNA2025トレーナー。

    - regions（マルチラベル）の場合は `sigmoid` を用いる。
    - multi-class の場合でも背景を含める。
    - `batch_dice` 設定は FocalTversky++ の `batch` に反映。
    - 係数はTversky同様に FNをやや重めに: alpha=0.3, beta=0.7。
    """

    def _build_loss(self):
        alpha = 0.3
        beta = 0.7
        gamma_pp = 2.0
        gamma_focal = 4.0 / 3.0  # 1.33...

        if self.label_manager.has_regions:
            # マルチラベル（領域）: sigmoid活性 + 背景込み
            loss = MemoryEfficientFocalTverskyPlusPlusLoss(
                alpha=alpha,
                beta=beta,
                gamma_pp=gamma_pp,
                gamma_focal=gamma_focal,
                apply_nonlin=torch.sigmoid,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                ddp=self.is_ddp,
            )
        else:
            # マルチクラス: softmax活性 + 背景込み
            loss = MemoryEfficientFocalTverskyPlusPlusLoss(
                alpha=alpha,
                beta=beta,
                gamma_pp=gamma_pp,
                gamma_focal=gamma_focal,
                apply_nonlin=softmax_helper_dim1,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                ddp=self.is_ddp,
            )

        # Deep Supervisionに対応
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # 解像度が低いほど重みを小さく（指数減衰）
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                # DDP時の特殊ケース: 最終出力に微小重み
                weights[-1] = 1e-6
            else:
                # 通常は最終出力を無視
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class FocalTverskyPlusPlusCELoss(nn.Module):
    """
    Focal Tversky++ と Cross Entropy を加重結合した損失。

    tversky_loss.py の TverskyPlusCELoss と同様に、ignore_label に応じた loss_mask によって
    FocalTversky++ 側をマスキングしつつ、CE 側は RobustCrossEntropyLoss に委譲します。
    """

    def __init__(
        self,
        tverskypp_kwargs: dict,
        ce_kwargs: dict,
        weight_tverskypp: float = 1.0,
        weight_ce: float = 1.0,
        ignore_label: int | None = None,
    ) -> None:
        super().__init__()
        self.weight_tverskypp = weight_tverskypp
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.tverskypp = MemoryEfficientFocalTverskyPlusPlusLoss(**tverskypp_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_label is not None:
            mask_bool = target != self.ignore_label
            mask = mask_bool.float()
            target_tversky = torch.clone(target)
            target_tversky[target == self.ignore_label] = 0
            num_fg = mask_bool.sum()
        else:
            mask = None
            target_tversky = target
            num_fg = None

        tverskypp_loss = self.tverskypp(net_output, target_tversky, loss_mask=mask)

        if self.ignore_label is None or (num_fg is not None and num_fg > 0):
            ce_loss = self.ce(net_output, target)
        else:
            ce_loss = torch.as_tensor(0.0, device=net_output.device, dtype=net_output.dtype)

        return self.weight_tverskypp * tverskypp_loss + self.weight_ce * ce_loss


class RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusPlusCE(RSNA2025Trainer_moreDAv3):
    """
    Focal Tversky++ と Cross Entropy を組み合わせたトレーナー（moreDAv3）。

    multi-class 専用（regions=True では BCE を利用すべき）。
    """

    def _build_loss(self):
        # multi-class 専用
        assert (
            not self.label_manager.has_regions
        ), "RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusPlusCEはmulti-class専用です (regionsではBCEをご利用ください)"

        tverskypp_kwargs = {
            "alpha": 0.3,
            "beta": 0.7,
            "gamma_pp": 2.0,
            "gamma_focal": 4.0 / 3.0,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth_nr": 1e-5,
            "smooth_dr": 1e-5,
            "do_bg": False,  # multi-classでは背景を除外
            "ddp": self.is_ddp,
            "apply_nonlin": softmax_helper_dim1,
        }

        ce_kwargs = {}
        if self.label_manager.ignore_label is not None:
            ce_kwargs["ignore_index"] = self.label_manager.ignore_label

        loss = FocalTverskyPlusPlusCELoss(
            tverskypp_kwargs=tverskypp_kwargs,
            ce_kwargs=ce_kwargs,
            weight_tverskypp=1.0,
            weight_ce=1.0,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusWithBackgroundPlusCE(RSNA2025Trainer_moreDAv3):
    """
    Focal Tversky++ と Cross Entropy を組み合わせたトレーナー（moreDAv3）。

    multi-class 専用（regions=True では BCE を利用すべき）。
    """

    def _build_loss(self):
        # multi-class 専用
        assert (
            not self.label_manager.has_regions
        ), "RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusPlusCEはmulti-class専用です (regionsではBCEをご利用ください)"

        tverskypp_kwargs = {
            "alpha": 0.3,
            "beta": 0.7,
            "gamma_pp": 2.0,
            "gamma_focal": 4.0 / 3.0,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth_nr": 1e-5,
            "smooth_dr": 1e-5,
            "do_bg": True,  # multi-classでは背景を除外
            "ddp": self.is_ddp,
            "apply_nonlin": softmax_helper_dim1,
        }

        ce_kwargs = {}
        if self.label_manager.ignore_label is not None:
            ce_kwargs["ignore_index"] = self.label_manager.ignore_label

        loss = FocalTverskyPlusPlusCELoss(
            tverskypp_kwargs=tverskypp_kwargs,
            ce_kwargs=ce_kwargs,
            weight_tverskypp=1.0,
            weight_ce=1.0,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
