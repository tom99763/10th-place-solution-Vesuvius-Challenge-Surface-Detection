"""
Tversky Loss実装を含むRSNA2025用のトレーナー

Tversky Lossは、偽陽性と偽陰性に異なる重み付けを可能にするDice Lossの一般化版です。
alpha, betaパラメータでFPとFNのバランスを調整できます。
alpha=beta=0.5の場合、Dice Lossと等価になります。
"""

from typing import Callable
import warnings

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv3 import RSNA2025Trainer_moreDAv3
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv5 import RSNA2025Trainer_moreDAv5
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class MemoryEfficientTverskyLoss(nn.Module):
    """
    Tversky Loss実装

    Tversky Index = TP / (TP + alpha*FP + beta*FN)

    Parameters:
        alpha: 偽陽性の重み (デフォルト: 0.3)
        beta: 偽陰性の重み (デフォルト: 0.7)
        smooth: ゼロ除算を避けるための平滑化項
        apply_nonlin: 活性化関数 (softmaxまたはsigmoid)
        batch_dice: バッチ単位での計算フラグ
        do_bg: 背景クラスを含むかどうか
        ddp: 分散学習対応フラグ
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
    ):
        super(MemoryEfficientTverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

        # alphaとbetaの和が1になることを推奨（必須ではない）
        if abs(self.alpha + self.beta - 1.0) > 0.01:
            warnings.warn(
                f"alpha ({alpha}) + beta ({beta}) = {alpha + beta} != 1.0. 通常、alpha + beta = 1.0に設定することが推奨されます",
                UserWarning,
            )

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # 次元を(b, c)の形状に変換
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # GTがすでにone-hotエンコーディングの場合
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            # True Positive + False Negative (GT)
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # 背景を除外する場合
        if not self.do_bg:
            x = x[:, 1:]

        # True Positive
        if loss_mask is None:
            tp = (x * y_onehot).sum(axes)
            # False Positive + True Positive (予測)
            sum_pred = x.sum(axes)
        else:
            tp = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        # False Positive = 予測 - TP
        fp = sum_pred - tp
        # False Negative = GT - TP
        fn = sum_gt - tp

        if self.ddp and self.batch_dice:
            # 分散学習時の集約
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        # Tversky Index計算
        if self.batch_dice:
            # バッチ方向で合計してからクラス平均（nnUNetのDiceと同一の集約）
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)
            # 依然としてバッチ次元が残るので合計
            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

            tversky_per_class = (tp + self.smooth) / torch.clip(
                tp + self.alpha * fp + self.beta * fn + self.smooth, min=1e-8
            )
            tversky = tversky_per_class.mean()
        else:
            # サンプルごと・クラスごとのTverskyを平均
            nominator = tp + self.smooth
            denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky = (nominator / torch.clip(denominator, min=1e-8)).mean()

        # 損失として返すため負値に
        return -tversky


class TverskyPlusCELoss(nn.Module):
    """
    Tversky LossとCross Entropy Lossの組み合わせ
    """

    def __init__(self, tversky_kwargs, ce_kwargs, weight_tversky=1, weight_ce=1, ignore_label=None):
        super().__init__()
        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.tversky = MemoryEfficientTverskyLoss(**tversky_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output, target):
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

        tversky_loss = self.tversky(net_output, target_tversky, loss_mask=mask)

        if self.ignore_label is None or (num_fg is not None and num_fg > 0):
            ce_loss = self.ce(net_output, target)
        else:
            ce_loss = torch.as_tensor(0.0, device=net_output.device, dtype=net_output.dtype)

        result = self.weight_tversky * tversky_loss + self.weight_ce * ce_loss
        return result


class RSNA2025Trainer_moreDAv3_TverskyLoss(RSNA2025Trainer_moreDAv3):
    """
    Tversky Lossを使用するRSNA2025トレーナー

    alphaとbetaパラメータで偽陽性と偽陰性の重みを調整できます。
    デフォルトでは偽陰性（見逃し）により大きなペナルティを課します。
    """

    def _build_loss(self):
        """Tversky Lossを構築"""
        tversky_alpha = 0.3
        tversky_beta = 0.7

        if self.label_manager.has_regions:
            # マルチラベル（領域）の場合はsigmoid活性化
            loss = MemoryEfficientTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                apply_nonlin=torch.sigmoid,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth=1e-5,
                ddp=self.is_ddp,
            )
        else:
            # マルチクラスの場合はsoftmax活性化
            loss = MemoryEfficientTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                apply_nonlin=softmax_helper_dim1,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=False,
                smooth=1e-5,
                ddp=self.is_ddp,
            )

        # Deep Supervisionの設定
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # 解像度が低くなるにつれて指数的に重みを減少させる
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                # DDPの特殊なケース
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # 重みを正規化
            weights = weights / weights.sum()

            # Deep Supervisionラッパーで包む
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class RSNA2025Trainer_moreDAv3_TverskyPlusCE(RSNA2025Trainer_moreDAv3):
    """
    Tversky LossとCross Entropy Lossを組み合わせたトレーナー
    """

    def _build_loss(self):
        """Tversky + CE複合損失を構築"""
        # このトレーナーはmulti-class専用（regionsではBCEを用いるべき）
        assert (
            not self.label_manager.has_regions
        ), "RSNA2025Trainer_moreDAv3_TverskyPlusCEはmulti-class専用です (regionsではBCEをご利用ください)"

        tversky_kwargs = {
            "alpha": 0.3,
            "beta": 0.7,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,  # multi-classでは背景を除外
            "ddp": self.is_ddp,
            "apply_nonlin": softmax_helper_dim1,
        }

        ce_kwargs = {}
        if self.label_manager.ignore_label is not None:
            ce_kwargs["ignore_index"] = self.label_manager.ignore_label

        loss = TverskyPlusCELoss(
            tversky_kwargs=tversky_kwargs,
            ce_kwargs=ce_kwargs,
            weight_tversky=1,
            weight_ce=1,
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


class RSNA2025Trainer_moreDAv5_TverskyPlusCE(RSNA2025Trainer_moreDAv5):
    """
    Tversky LossとCross Entropy Lossを組み合わせたトレーナー
    """

    def _build_loss(self):
        """Tversky + CE複合損失を構築"""
        # このトレーナーはmulti-class専用（regionsではBCEを用いるべき）
        assert (
            not self.label_manager.has_regions
        ), "RSNA2025Trainer_moreDAv5_TverskyPlusCEはmulti-class専用です (regionsではBCEをご利用ください)"

        tversky_kwargs = {
            "alpha": 0.3,
            "beta": 0.7,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,  # multi-classでは背景を除外
            "ddp": self.is_ddp,
            "apply_nonlin": softmax_helper_dim1,
        }

        ce_kwargs = {}
        if self.label_manager.ignore_label is not None:
            ce_kwargs["ignore_index"] = self.label_manager.ignore_label

        loss = TverskyPlusCELoss(
            tversky_kwargs=tversky_kwargs,
            ce_kwargs=ce_kwargs,
            weight_tversky=1,
            weight_ce=1,
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
