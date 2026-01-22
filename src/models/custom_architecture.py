import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

TEMPLATE_CODES_3D = [
    # straight lines (axis-aligned)
    2**12 + 2**13,        # z line
    2**10 + 2**16,        # y line
    2**14 + 2**11,        # x line

    # endpoints
    2**12, 2**13,
    2**10, 2**16,
    2**14, 2**11,

    # simple junction
    2**12 + 2**10 + 2**14,
]

def make_3d_bit_kernel():
    """
    Returns a (1,1,3,3,3) kernel with unique bit weights
    for all 26 neighbors, center = 0.
    """
    kernel = torch.zeros(3, 3, 3, dtype=torch.float32)

    bit = 0
    for z in range(3):
        for y in range(3):
            for x in range(3):
                if (z, y, x) == (1, 1, 1):
                    continue
                kernel[z, y, x] = 2 ** bit
                bit += 1

    return kernel.view(1, 1, 3, 3, 3)


class StructureCode3D(nn.Module):
    """
    Soft 3D topology encoder using 3×3×3 bit stencils.
    """
    def __init__(self, template_codes, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        kernel = make_3d_bit_kernel()
        self.register_buffer("kernel", kernel)

        codes = torch.tensor(template_codes, dtype=torch.float32)
        self.register_buffer(
            "templates",
            codes.view(1, -1, 1, 1, 1)
        )

    def forward(self, prob):
        """
        prob: (B,1,D,H,W)
        returns: (B,K,D,H,W)
        """
        if prob.shape[1] > 1:
            prob = prob[:, :1]

        # soft binarization
        w = torch.sigmoid((prob - 0.5) / self.temperature)

        # 3D pattern code
        code = F.conv3d(w, self.kernel, padding=1)

        # soft equality to templates
        sim = torch.exp(-(code - self.templates).abs())

        return sim


class CustomUNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        base_channels=(32, 64, 128, 256, 320, 320),
    ):
        super().__init__()

        self.backbone = ResidualEncoderUNet(
            input_channels=in_channels,
            n_stages=len(base_channels),
            features_per_stage=base_channels,
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=(1, 2, 2, 2, 2, 2),
            n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
            num_classes=out_channels,
            n_conv_per_stage_decoder=[1] * (len(base_channels) - 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )

        self.structure_enc = StructureCode3D(TEMPLATE_CODES_3D)

        # Project structure channels to match encoder skip channels
        skip_channels = base_channels[:-1]  # encoder skips
        struct_channels = len(TEMPLATE_CODES_3D)
        self.struct_projections = nn.ModuleList([
            nn.Conv3d(struct_channels, c, kernel_size=1)
            for c in skip_channels
        ])

    def forward(self, x, prob_map):
        """
        x        : (B,C,D,H,W)
        prob_map : (B,1,D,H,W)
        """
        skips = self.backbone.encoder(x)
        new_skips = []

        for i, skip in enumerate(skips):
            _, _, D, H, W = skip.shape

            # Resize probability map to match skip resolution
            prob_resized = F.interpolate(
                prob_map,
                size=(D, H, W),
                mode="trilinear",
                align_corners=False
            )

            # Compute structure codes
            struct = self.structure_enc(prob_resized)

            # Project structure channels to match skip channels
            struct_proj = self.struct_projections[i](struct)

            # Safe concatenation
            new_skip = skip + struct_proj
            new_skips.append(new_skip)

        # Pass to decoder
        return self.backbone.decoder(new_skips)