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


class ProgressiveEncoder3D(nn.Module):
    def __init__(self, latent_dim = 256, base_channels=256, max_steps = 3):
        super().__init__()
        '''
        * control max steps 
        '''
        self.from_voxel = nn.ModuleList([nn.Conv3d(1, base_channels//2**i, 3, 1, 1) for i in reversed(range(1,  max_steps + 2))])

        # Progressive downsampling blocks
        self.blocks = nn.ModuleList(reversed([
            nn.Sequential(
                nn.Conv3d(base_channels//2 ** (i+1), base_channels//2 ** i, 4, 2, 1),
                nn.BatchNorm3d(base_channels//2 ** i),
                nn.LeakyReLU(0.2)
            )
            for i in range(max_steps + 1)
        ]))

        self.from_prev_blocks = nn.ModuleList(reversed([
            nn.Conv3d(base_channels//2 ** (i+1), base_channels//2 ** i, 1) for i in range(max_steps + 1)
        ]))

        self.to_latent_list =  nn.ModuleList(reversed([
            nn.Conv3d(base_channels//2**i, latent_dim, 1) for i in range(max_steps + 1)
        ]))

        self.to_mu = nn.Conv3d(base_channels, latent_dim, 1)
        self.to_logvar = nn.Conv3d(base_channels, latent_dim, 1)

    def forward(self, x, step=0, alpha=0.5):
        x = self.from_voxel[min(step-1, 0)](x)

        if step == 0:
            x = self.blocks[-1](x)
            patch_out = self.to_latent_list[-1](x)
        else:
            for s in range(step + 1):
                x_prev = x
                x = self.blocks[s](x)
                x = alpha * x + (1-alpha) * F.avg_pool3d(self.from_prev_blocks[s](x_prev), kernel_size=2)
            patch_out = self.to_latent_list[s](x)
        mu = self.to_mu(patch_out)
        logvar = self.to_logvar(patch_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class ProgressiveGenerator3D(nn.Module):
    def __init__(self, latent_dim=256, base_channels=256, max_steps=3):
        super().__init__()
        self.max_steps = max_steps

        # Project latent → initial 3D feature map
        self.from_latent = nn.Conv3d(latent_dim, base_channels, 1)

        # Progressive blocks (max_steps + 1, same as encoder)
        ch = base_channels
        self.blocks = nn.ModuleList()
        self.to_voxel = nn.ModuleList()

        for i in range(max_steps + 1):
            next_ch = ch // 2

            self.blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(ch, next_ch, 4, 2, 1),
                    nn.BatchNorm3d(next_ch),
                    nn.ReLU(inplace=True)
                )
            )

            self.to_voxel.append(nn.Conv3d(next_ch, 1, 1))

            ch = next_ch

        # old-resolution upsampling for fade-in
        self.upsamplers = nn.ModuleList([
            nn.Conv3d(base_channels // (2**i), base_channels // (2**(i+1)), 1)
            for i in range(max_steps)
        ])

    def forward(self, z, step=0, alpha=1.0):
        # z: [B, latent_dim, D, H, W]
        x = self.from_latent(z)

        if step == 0:
            x = self.blocks[0](x)
            return torch.sigmoid(self.to_voxel[0](x))

        for s in range(step + 1):

            x_new = self.blocks[s](x)

            # Fade-in only at highest current step
            if s == step:
                # Old-resolution output
                x_old = self.to_voxel[s - 1](x)
                x_old = F.interpolate(
                    x_old,
                    scale_factor=2,
                    mode="trilinear",
                    align_corners=False
                )

                out = (1 - alpha) * x_old + alpha * self.to_voxel[s](x_new)
                return torch.sigmoid(out)

            x = x_new


if __name__ == '__main__':
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Progressive parameters
    max_steps = 3
    step = 3
    alpha = 0.5

    # Models
    encoder = ProgressiveEncoder3D(max_steps=max_steps).to(device)
    generator = ProgressiveGenerator3D(max_steps=max_steps).to(device)

    # Example input: batch of 1, 1 channel, 256³ volume
    x = torch.randn(1, 1, 256, 256, 256).to(device)
    print("Input x shape:", x.shape)

    with torch.no_grad():

        # --- Encoder ---
        mu, logvar = encoder(x, step=step)
        print("mu shape:", mu.shape)
        print("logvar shape:", logvar.shape)

        z = encoder.reparameterize(mu, logvar)
        print("Latent z shape:", z.shape)

        # --- Generator ---
        x_hat = generator(z, step=step, alpha=alpha)
        print("Reconstructed volume shape:", x_hat.shape)

    print("Done.")

