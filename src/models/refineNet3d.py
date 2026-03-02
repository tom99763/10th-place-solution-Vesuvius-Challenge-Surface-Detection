# Retry with corrected UNet upsampling (uses trilinear interpolation to avoid size mismatches).
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from src.models.custom_architecture import CustomUNet

def create_residual_unet(
        in_channels=2,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(1, 2, 2, 2),
        n_blocks_per_stage=(1, 2, 2, 2),
        deep_supervision=False,
):
    n_conv_per_stage_decoder = [1] * (len(channels) - 1)

    model = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=len(channels),
        features_per_stage=channels,
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=out_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    return model


def create_residual_unet_small(
        in_channels=2,
        out_channels=3,
        channels=(16, 32, 64, 128, 160, 160),
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 2, 2, 3, 3, 3),
        deep_supervision=False,
):
    n_conv_per_stage_decoder = [1] * (len(channels) - 1)

    model = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=len(channels),
        features_per_stage=channels,
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=out_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    return model

class RefineDynUnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = create_residual_unet(
            in_channels=3,
            out_channels=2
        )
    def forward(self, x):
       return self.predictor(x)