from typing import List, Tuple, Union

import torch
from monai.networks.nets import SwinUNETR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.nnUNetTrainerSkeletonRecall_more_DAv3 import (
    nnUNetTrainerSkeletonRecall_more_DAv3,
)

from src.models.schedulers.warmup_cosine_annealing import CosineAnnealingLR as WarmupCosineAnnealingLR


SWIN_DEFAULT_FEATURE_SIZE = 24


def _compute_warmup_params(num_epochs: int, initial_lr: float) -> Tuple[int, float, float]:
    """ウォームアップ関連のハイパーパラメータをまとめて計算する。"""
    if num_epochs > 1:
        if num_epochs >= 10:
            warmup_epochs = max(5, min(50, num_epochs // 10))
        else:
            warmup_epochs = max(1, num_epochs // 2)
        warmup_epochs = min(warmup_epochs, num_epochs - 1)
        warmup_start_lr = initial_lr * 1e-2
    else:
        warmup_epochs = 0
        warmup_start_lr = initial_lr
    min_lr = initial_lr * 1e-2
    return warmup_epochs, warmup_start_lr, min_lr


class RSNA2025TrainerSwinUNETR(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 8e-4
        self.enable_deep_supervision = False
        # ウォームアップ期間をエポック数に応じて設定
        self.warmup_epochs, self.warmup_start_lr, self.min_lr = _compute_warmup_params(
            self.num_epochs, self.initial_lr
        )
        self.swin_feature_size = SWIN_DEFAULT_FEATURE_SIZE

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        _ = architecture_class_name
        _ = arch_init_kwargs
        _ = arch_init_kwargs_req_import
        model = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            feature_size=self.swin_feature_size,
            use_checkpoint=False,
            spatial_dims=len(self.configuration_manager.patch_size),
            use_v2=True,
        )
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.num_epochs,
            warmup_start_lr=self.warmup_start_lr,
            eta_min=self.min_lr,
        )
        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        # SwinUNETRはDeep Supervision非対応のため無効化
        return


class RSNA2025TrainerSkeletonRecallSwinUNETR_moreDAv3(nnUNetTrainerSkeletonRecall_more_DAv3):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 8e-4
        self.enable_deep_supervision = False
        # ウォームアップ期間をエポック数に応じて設定
        self.warmup_epochs, self.warmup_start_lr, self.min_lr = _compute_warmup_params(
            self.num_epochs, self.initial_lr
        )
        self.swin_feature_size = SWIN_DEFAULT_FEATURE_SIZE

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        _ = architecture_class_name
        _ = arch_init_kwargs
        _ = arch_init_kwargs_req_import
        model = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            feature_size=self.swin_feature_size,
            use_checkpoint=False,
            spatial_dims=len(self.configuration_manager.patch_size),
            use_v2=True,
        )
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.num_epochs,
            warmup_start_lr=self.warmup_start_lr,
            eta_min=self.min_lr,
        )
        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        # SwinUNETRはDeep Supervision非対応のため無効化
        return
