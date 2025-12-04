'''Reference
* Park, S., & Kim, H. (2021). Facevae: Generation of a 3d geometric object using variational autoencoders. Electronics, 10(22), 2792.
* https://github.com/agrija9/Convolutional-VAE-for-3D-Turbulence-Data
* https://github.com/IsaacGuan/3D-VAE
* https://github.com/Spartey/3D-VAE-GAN-Deep-Learning-Project (*****)
* https://ar5iv.labs.arxiv.org/html/1912.08283
'''

#Coarse-to-Fine Progressive 3D VAEGAN for 3D Scroll segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.grammar import cfg_demo


class ProgressiveEncoder3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        latent_dim = self.cfg.latent_dim
        base_channels = self.cfg.base_channels
        max_steps = self.cfg.max_steps
        self.from_voxel = nn.ModuleList([
            nn.Conv3d(1, base_channels//2**(i+1), 3, 1, 1) for i in reversed(range(max_steps + 1))])

        # Progressive downsampling blocks
        self.blocks = nn.ModuleList(
            (reversed([
                nn.Sequential(
                    nn.Conv3d(base_channels // 2 ** (i + 1), base_channels // 2 ** i, 4, 2, 1),
                    nn.BatchNorm3d(base_channels // 2 ** i),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(base_channels // 2 ** i, base_channels // 2 ** i, 4, 2, 1),
                    nn.BatchNorm3d(base_channels // 2 ** i),
                    nn.LeakyReLU(0.2)
                ) if i == 0 else
                nn.Sequential(
                    nn.Conv3d(base_channels // 2 ** (i + 1), base_channels // 2 ** i, 4, 2, 1),
                    nn.BatchNorm3d(base_channels // 2 ** i),
                    nn.LeakyReLU(0.2)
                )
                for i in range(max_steps + 1)
            ]))
        )

        self.from_prev_blocks = nn.ModuleList(reversed([
            nn.Conv3d(base_channels//2 ** (i+1), base_channels//2 ** i, 1) for i in range(max_steps + 1)
        ]))
        self.to_mu = nn.Conv3d(base_channels, latent_dim, 1)
        self.to_logvar = nn.Conv3d(base_channels, latent_dim, 1)

    def forward(self, x, step=0, alpha=0.5):
        x = self.from_voxel[-step-1](x)

        if step == 0:
            x = self.blocks[-step-1](x)
        else:
            for s in reversed(range(step + 1)):
                x_prev = x
                x = self.blocks[-s-1](x)
                x = alpha * x + (1-alpha) * F.avg_pool3d(self.from_prev_blocks[-s-1](x_prev), kernel_size = 4 if s==0 else 2)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class PositionalEncoding3D(nn.Module):
    def __init__(self, num_bands=6, include_input=True):
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input

        # Frequencies: 1, 2, 4, ..., 2^(num_bands-1)
        self.freq_bands = 2 ** torch.arange(num_bands).float()

    def forward(self, D, H, W, device):
        """
        Returns tensor of shape [C_pe, D, H, W]
        """

        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)

        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([zz, yy, xx], dim=0)  # [3, D, H, W]

        pe_list = []
        if self.include_input:
            pe_list.append(coords)

        for freq in self.freq_bands.to(device):
            for func in (torch.sin, torch.cos):
                pe_list.append(func(coords * freq))

        return torch.cat(pe_list, dim=0)  # [C_pe, D, H, W]


class ProgressiveGenerator3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        latent_dim = cfg.latent_dim
        base_channels = cfg.base_channels
        max_steps = cfg.max_steps
        pe_bands = cfg.pe_bands
        self.max_steps = max_steps

        # Positional encoding module
        self.pos_encoding = PositionalEncoding3D(num_bands=pe_bands)

        # Channels introduced by PE
        pe_channels = 3 * (1 + 2 * pe_bands)

        # Project latent → feature map
        self.from_latent = nn.Conv3d(latent_dim, base_channels, 1)

        # Combine latent features + PE
        self.pe_proj = nn.Conv3d(base_channels + pe_channels, base_channels, 1)

        # Progressive blocks
        ch = base_channels
        self.blocks = nn.ModuleList()
        self.to_voxel = nn.ModuleList()

        for i in range(max_steps + 1):
            next_ch = ch // 2
            if i ==0:
                self.blocks.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(ch, ch, 4, 2, 1),
                        nn.BatchNorm3d(ch),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose3d(ch, next_ch, 4, 2, 1),
                        nn.BatchNorm3d(next_ch),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.blocks.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(ch, next_ch, 4, 2, 1),
                        nn.BatchNorm3d(next_ch),
                        nn.ReLU(inplace=True)
                    )
                )

            self.to_voxel.append(nn.Conv3d(next_ch, 1, 1))

            ch = next_ch

        # fade-in upsamplers (unchanged)
        self.upsamplers = nn.ModuleList([
            nn.Conv3d(base_channels // (2**i), base_channels // (2**(i+1)), 1)
            for i in range(max_steps)
        ])

    def forward(self, z, step=0, alpha=1.0):
        B, C, D, H, W = z.shape

        x = self.from_latent(z)

        # --- Add positional encoding ---
        pe = self.pos_encoding(D, H, W, device=z.device)  # [C_pe, D, H, W]
        pe = pe.unsqueeze(0).expand(B, -1, -1, -1, -1)
        x = torch.cat([x, pe], dim=1)
        x = self.pe_proj(x)
        # --------------------------------

        if step == 0:
            x = self.blocks[step](x)
            return torch.sigmoid(self.to_voxel[step](x))

        for s in range(step + 1):

            x_new = self.blocks[s](x)

            if s == step:
                x_old = self.to_voxel[s - 1](x)
                x_old = F.interpolate(
                    x_old, scale_factor=2,
                    mode="trilinear", align_corners=False
                )

                out = (1 - alpha) * x_old + alpha * self.to_voxel[s](x_new)
                return torch.sigmoid(out)

            x = x_new


if __name__ == '__main__':
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)


    class CFG:
        def __init__(self):
            # Latent vector dimensionality
            self.latent_dim = 128
            self.base_channels = 256
            self.max_steps = 3  # used in encoder
            self.pe_bands = 6  # 3*(1+2*bands) extra channels
            self.start_resolution = 32
            self.alpha = 0.5

    cfg = CFG()
    encoder = ProgressiveEncoder3D(cfg).to(device)
    generator = ProgressiveGenerator3D(cfg).to(device)
    step = 1

    # Example input: batch of 1, 1 channel, 256³ volume
    x = torch.randn(1, 1, 64, 64, 64).to(device)
    print("Input x shape:", x.shape)

    with torch.no_grad():

        # --- Encoder ---
        mu, logvar = encoder(x, step=step)
        print("mu shape:", mu.shape)
        print("logvar shape:", logvar.shape)

        z = encoder.reparameterize(mu, logvar)
        print("Latent z shape:", z.shape)

        # --- Generator ---
        x_hat = generator(z, step=step, alpha=cfg.alpha)
        print("Reconstructed volume shape:", x_hat.shape)

    print("Done.")

