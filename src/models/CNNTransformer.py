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
    def __init__(self, dim, depth=4, window_size=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            WindowAttention3D(dim, window_size)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(x)
        return x


# class UNetWithTransformer(nn.Module):
#     def __init__(
#         self,
#         in_channels=2,
#         out_channels=4,
#         base_channels=(32,64,128,256,320,320),
#         strides=(1,2,2,2,2,2),
#         n_blocks=(1,3,4,6,6,6),
#         transformer_depth=2,
#         window_size=4,
#         max_v=1.0,
#     ):
#         super().__init__()
#
#         self.backbone = ResidualEncoderUNet(
#             input_channels=in_channels,
#             n_stages=len(base_channels),
#             features_per_stage=base_channels,
#             conv_op=nn.Conv3d,
#             kernel_sizes=3,
#             strides=strides,
#             n_blocks_per_stage=n_blocks,
#             num_classes=out_channels,
#             n_conv_per_stage_decoder=[1] * (len(base_channels) - 1),
#             conv_bias=True,
#             norm_op=nn.InstanceNorm3d,
#             nonlin=nn.LeakyReLU,
#             nonlin_kwargs={"inplace": True},
#             deep_supervision=False,
#         )
#
#         self.transformer = BottleneckTransformer3D(
#             dim=base_channels[-1],
#             depth=transformer_depth,
#             window_size=window_size,
#         )
#
#         self.max_v = max_v
#
#     def forward(self, x):
#         # --- encoder ---
#         skips = self.backbone.encoder(x)
#
#         # --- transformer at bottleneck ---
#         skips[-1] = self.transformer(skips[-1])
#
#         # --- decoder ---
#         output = self.backbone.decoder(skips)
#         return output



class UNetWithTransformer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        base_channels=(32, 64, 128, 256, 320, 320),
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks=(1, 3, 4, 6, 6, 6),
        transformer_indices=(-4, -3, -2, -1),
        transformer_depth=2,
        window_size=4,
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

        self.transformer_indices = set(transformer_indices)

        # one transformer per selected skip
        self.transformers = nn.ModuleDict()
        for idx in self.transformer_indices:
            ch = base_channels[idx]
            self.transformers[str(idx)] = BottleneckTransformer3D(
                dim=ch,
                depth=transformer_depth,
                window_size=window_size,
            )

        '''
        surface_indices = (-6, -5)  # example, highest resolutions
        self.surface_blocks = nn.ModuleDict({
            str(idx): SurfaceAwareConv3D(base_channels[idx])
            for idx in surface_indices
        })
        '''

    def forward(self, x):
        skips = self.backbone.encoder(x)

        for idx in self.transformer_indices:
            skips[idx] = self.transformers[str(idx)](skips[idx])

        '''
        for idx in surface_indices:
            skips[idx] = self.surface_blocks[str(idx)](skips[idx])
        '''

        output = self.backbone.decoder(skips)
        return output



class SurfaceAwareConv3D(nn.Module):
    """
    Surface-aware convolution for shallow skip connections.
    Improves boundary alignment (SurfaceDice@τ) without affecting topology.
    """
    def __init__(
        self,
        channels,
        grad_channels=None,
        kernel_size=3,
        use_norm=True,
        use_residual=True,
    ):
        super().__init__()
        self.channels = channels
        self.use_residual = use_residual

        if grad_channels is None:
            grad_channels = channels // 2

        padding = kernel_size // 2

        # Base local appearance conv
        self.conv = nn.Conv3d(
            channels, channels, kernel_size,
            padding=padding, bias=False
        )

        # Gradient fusion (surface normal encoding)
        self.grad_fuse = nn.Conv3d(
            4, grad_channels, kernel_size=1, bias=False
        )

        # Merge appearance + geometry
        self.merge = nn.Conv3d(
            channels + grad_channels, channels, kernel_size=1, bias=False
        )

        self.norm = nn.InstanceNorm3d(channels) if use_norm else nn.Identity()
        self.act = nn.LeakyReLU(inplace=True)

        # Register fixed gradient kernels (not learnable)
        self.register_buffer("sobel_x", self._sobel_kernel(axis=0))
        self.register_buffer("sobel_y", self._sobel_kernel(axis=1))
        self.register_buffer("sobel_z", self._sobel_kernel(axis=2))

    def _sobel_kernel(self, axis):
        """
        Create 3D Sobel-like kernel for x, y, or z gradient
        """
        k = torch.tensor([-1, 0, 1], dtype=torch.float32)
        s = torch.tensor([1, 2, 1], dtype=torch.float32)

        if axis == 0:  # x
            kernel = k[:, None, None] * s[None, :, None] * s[None, None, :]
        elif axis == 1:  # y
            kernel = s[:, None, None] * k[None, :, None] * s[None, None, :]
        else:  # z
            kernel = s[:, None, None] * s[None, :, None] * k[None, None, :]

        kernel = kernel / kernel.abs().sum()
        return kernel[None, None]  # (1,1,3,3,3)

    def compute_gradients(self, x):
        """
        Compute per-voxel gradient magnitude and directions
        """
        B, C, D, H, W = x.shape

        # Average across channels for geometry
        x_gray = x.mean(dim=1, keepdim=True)

        gx = F.conv3d(x_gray, self.sobel_x, padding=1)
        gy = F.conv3d(x_gray, self.sobel_y, padding=1)
        gz = F.conv3d(x_gray, self.sobel_z, padding=1)

        grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-6)

        return torch.cat([gx, gy, gz, grad_mag], dim=1)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        residual = x

        # Appearance branch
        feat = self.conv(x)

        # Geometry branch
        grads = self.compute_gradients(x)
        geom = self.grad_fuse(grads)

        # Merge
        out = self.merge(torch.cat([feat, geom], dim=1))
        out = self.norm(out)
        out = self.act(out)

        if self.use_residual:
            out = out + residual

        return out