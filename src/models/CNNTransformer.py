import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        ws = self.window_size

        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        _, _, Dp, Hp, Wp = x.shape

        x = rearrange(
            x,
            "b c (d ws1) (h ws2) (w ws3) -> (b d h w) (ws1 ws2 ws3) c",
            ws1=ws, ws2=ws, ws3=ws,
        )

        x = self.norm(x)
        x, _ = self.attn(x, x, x)

        x = rearrange(
            x,
            "(b d h w) (ws1 ws2 ws3) c -> b c (d ws1) (h ws2) (w ws3)",
            b=B,
            d=Dp // ws,
            h=Hp // ws,
            w=Wp // ws,
            ws1=ws, ws2=ws, ws3=ws,
        )

        # remove padding
        x = x[:, :, :D, :H, :W]
        return x


class BottleneckTransformer3D(nn.Module):
    def __init__(self, dim, depth=2, window_size=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            WindowAttention3D(dim, window_size)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(x)
        return x


class UNetWithTransformer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        base_channels=(32,64,128,256,320,320),
        strides=(1,2,2,2,2,2),
        n_blocks=(1,3,4,6,6,6),
        transformer_depth=2,
        window_size=4,
        max_v=1.0,
    ):
        super().__init__()

        self.backbone = ResidualEncoderUNet(
            input_channels=in_channels,
            n_stages=len(base_channels),
            features_per_stage=base_channels,
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=strides,
            n_blocks_per_stage=n_blocks,
            num_classes=out_channels,
            n_conv_per_stage_decoder=[1] * (len(base_channels) - 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )

        self.transformer = BottleneckTransformer3D(
            dim=base_channels[-1],
            depth=transformer_depth,
            window_size=window_size,
        )

        self.max_v = max_v

    def forward(self, x):
        # --- encoder ---
        skips = self.backbone.encoder(x)

        # --- transformer at bottleneck ---
        skips[-1] = self.transformer(skips[-1])

        # --- decoder ---
        output = self.backbone.decoder(skips)
        return output