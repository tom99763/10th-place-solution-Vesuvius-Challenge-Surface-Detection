import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class CustomUNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        base_channels=(32, 64, 128, 256, 320, 320),
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks=(1, 3, 4, 6, 6, 6),
        surf_indices=(-6, -5),
        mid_indices=(-4, -3),
        deep_indices=(-2, -1),
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

        self.surf_indices = surf_indices
        self.mid_indices = mid_indices
        self.deep_indices = deep_indices

        # --- skip processors ---
        self.surface_blocks = nn.ModuleDict({
            str(i): SurfaceAwareConv3D(base_channels[i])
            for i in surf_indices
        })

        self.mid_blocks = nn.ModuleDict({
            str(i): nn.Sequential(
                DeformableMidSkip3D(base_channels[i]),
                TopologyGatedSkip3D(base_channels[i])
            )
            for i in mid_indices
        })

        self.deep_blocks = nn.ModuleDict({
            str(i): nn.Sequential(
                LargeKernelSkip3D(base_channels[i]),
                TopologyGatedSkip3D(base_channels[i])
            )
            for i in deep_indices
        })

        # --- skip confidence ---
        self.skip_confidence = SkipConfidenceWeighting(len(base_channels))

    def forward(self, x):
        skips = self.backbone.encoder(x)

        for i in self.surf_indices:
            skips[i] = self.surface_blocks[str(i)](skips[i])

        for i in self.mid_indices:
            skips[i] = self.mid_blocks[str(i)](skips[i])

        for i in self.deep_indices:
            skips[i] = self.deep_blocks[str(i)](skips[i])

        skips = self.skip_confidence(skips)

        return self.backbone.decoder(skips)


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


class GatedSkip3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)

class LargeKernelSkip3D(nn.Module):
    def __init__(self, channels, k=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, k, padding=k//2, groups=channels),
            nn.Conv3d(channels, channels, 1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.block(x)


class TopologyGatedSkip3D(nn.Module):
    """
    Modulates skip strength based on topology stability proxies.
    """
    def __init__(self, channels, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        mean = x.mean(dim=(1,2,3,4))
        var = x.var(dim=(1,2,3,4))
        grad = torch.mean(torch.abs(x[..., 1:] - x[..., :-1]), dim=(1,2,3,4))

        stats = torch.stack([mean, var, grad], dim=1)  # (B,3)
        gate = self.mlp(stats).view(-1,1,1,1,1)

        return x * gate

class SkipConfidenceWeighting(nn.Module):
    """
    Learns relative importance of skip levels.
    """
    def __init__(self, n_skips):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_skips))

    def forward(self, skips):
        weights = torch.softmax(self.logits, dim=0)
        return [w * s for w, s in zip(weights, skips)]


class DeformableMidSkip3D(nn.Module):
    """
    Mid-resolution deformable skip using feature warping.
    Topology-safe and stable for thin sheets.
    """
    def __init__(self, channels):
        super().__init__()

        self.flow_net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, 3, 3, padding=1),
            nn.Tanh()
        )

        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        flow = self.flow_net(x)  # (B, 3, D, H, W)

        # ---- build base grid (no grad needed) ----
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, D, device=x.device),
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij"
        )
        base_grid = torch.stack((grid_w, grid_h, grid_d), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        # ---- normalize flow WITHOUT inplace ops ----
        flow = flow.permute(0, 2, 3, 4, 1)  # (B,D,H,W,3)

        scale = torch.tensor(
            [W, H, D],
            device=x.device,
            dtype=flow.dtype
        )

        flow_norm = flow / scale  # <-- out-of-place

        warped = F.grid_sample(
            x,
            base_grid + flow_norm,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return self.refine(warped)