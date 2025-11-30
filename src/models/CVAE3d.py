'''Reference
* Park, S., & Kim, H. (2021). Facevae: Generation of a 3d geometric object using variational autoencoders. Electronics, 10(22), 2792.
* https://github.com/agrija9/Convolutional-VAE-for-3D-Turbulence-Data
* https://github.com/IsaacGuan/3D-VAE
* https://github.com/Spartey/3D-VAE-GAN-Deep-Learning-Project (*****)
'''

#Coarse-to-Fine Progressive 3D VAEGAN for 3D Scroll segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=256, base_channels=32):
        super().__init__()
        # Progressive downsampling
        self.conv1 = nn.Conv3d(in_channels, base_channels, 4, 2, 1)      # 256 -> 128
        self.conv2 = nn.Conv3d(base_channels, base_channels*2, 4, 2, 1)  # 128 -> 64
        self.conv3 = nn.Conv3d(base_channels*2, base_channels*4, 4, 2, 1) # 64 -> 32
        self.conv4 = nn.Conv3d(base_channels*4, base_channels*8, 4, 2, 1) # 32 -> 16
        self.conv5 = nn.Conv3d(base_channels*8, base_channels*16, 4, 2, 1) # 16 -> 8
        self.conv6 = nn.Conv3d(base_channels*16, base_channels*32, 4, 2, 1) # 8 -> 4

        self.fc_mu = nn.Linear(base_channels*32*4*4*4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels*32*4*4*4, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class ProgressiveGenerator3D(nn.Module):
    def __init__(self, base_channels=256):
        super().__init__()
        self.init_size = 16

        # Progressive blocks for 32³ → 256³
        self.blocks = nn.ModuleList([
            nn.Sequential( # 32³
                nn.ConvTranspose3d(base_channels, base_channels//2, 4, 2, 1),
                nn.BatchNorm3d(base_channels//2),
                nn.ReLU()
            ),
            nn.Sequential( # 64³
                nn.ConvTranspose3d(base_channels//2, base_channels//4, 4, 2, 1),
                nn.BatchNorm3d(base_channels//4),
                nn.ReLU()
            ),
            nn.Sequential( # 128³
                nn.ConvTranspose3d(base_channels//4, base_channels//8, 4, 2, 1),
                nn.BatchNorm3d(base_channels//8),
                nn.ReLU()
            ),
            nn.Sequential( # 256³
                nn.ConvTranspose3d(base_channels//8, base_channels//16, 4, 2, 1),
                nn.BatchNorm3d(base_channels//16),
                nn.ReLU()
            )
        ])

        self.to_next_blocks = nn.ModuleList([
            nn.ConvTranspose3d(base_channels, base_channels // 2, 1),
            nn.ConvTranspose3d(base_channels//2, base_channels // 4, 1),
            nn.ConvTranspose3d(base_channels//4, base_channels // 8, 1),
            nn.ConvTranspose3d(base_channels//8, base_channels // 16, 1)
        ])
        self.to_voxel = nn.ModuleList([
            nn.Conv3d(base_channels//2, 1, 1),
            nn.Conv3d(base_channels // 4, 1, 1),
            nn.Conv3d(base_channels // 8, 1, 1),
            nn.Conv3d(base_channels // 16, 1, 1)
        ])

    def forward(self, z, step=0, alpha=0.5):
        x = z[..., None, None, None].repeat(1, 1, self.init_size, self.init_size, self.init_size)
        if step == 0:
            x = self.blocks[step](x)
            return torch.sigmoid(self.to_voxel[step](x))
        else:
            for s in range(step + 1):
                x_prev = x
                x = self.blocks[s](x)
                x = alpha * x + (1-alpha) * F.interpolate(self.to_next_blocks[s](x_prev), scale_factor=2, mode='trilinear', align_corners=False)
            return torch.sigmoid(self.to_voxel[s](x))


class ProgressiveDiscriminator3D(nn.Module):
    def __init__(self, base_channels=256):
        super().__init__()
        self.from_voxel = nn.ModuleList([nn.Conv3d(1, base_channels//2**i, 3, 1, 1) for i in range(4)])

        # Progressive downsampling blocks
        self.blocks = nn.ModuleList(reversed([
            nn.Sequential(  # 64³ -> 32³
                nn.Conv3d(base_channels//2, base_channels, 4, 2, 1),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(  # 128³ -> 64³
                nn.Conv3d(base_channels//4, base_channels//2, 4, 2, 1),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(  # 256³ -> 128³
                nn.Conv3d(base_channels//8, base_channels//4, 4, 2, 1),
                nn.LeakyReLU(0.2)
            )
        ]))

        # 1x1x1 conv to map previous resolution features for fade-in
        self.from_prev_blocks = nn.ModuleList(reversed([
            nn.Conv3d(base_channels//2, base_channels, 1),
            nn.Conv3d(base_channels//4, base_channels//2, 1),
            nn.Conv3d(base_channels//8, base_channels//4, 1)
        ]))

        # Final 1x1x1 conv instead of Linear for PatchGAN output
        self.to_critic_list =  nn.ModuleList([
            nn.Conv3d(base_channels//4, 1, 1),
            nn.Conv3d(base_channels//8, 1, 1),
            nn.Conv3d(base_channels, 1, 1),
        ])

    def forward(self, x, step=0, alpha=0.5):
        x = self.from_voxel[step](x)

        if step == 0:
            x = self.blocks[step](x)
            patch_out = self.to_critic_list[step](x)
        else:
            for s in range(step):
                x_prev = x
                x = self.blocks[s](x)
                x = alpha * x + (1-alpha) * F.avg_pool3d(self.from_prev_blocks[s](x_prev), kernel_size=2)
            patch_out = self.to_critic_list[s](x)
        return patch_out


if __name__ == '__main__':
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate models
    latent_dim = 256
    encoder = Encoder3D(in_channels=1, latent_dim=latent_dim).to(device)
    generator = ProgressiveGenerator3D().to(device)
    discriminator = ProgressiveDiscriminator3D().to(device)

    # Example input: batch of 1, 1 channel, 256³ volume
    x = torch.randn(1, 1, 256, 256, 256).to(device)

    # --- Forward pass through encoder ---
    mu, logvar = encoder(x)
    z = encoder.reparameterize(mu, logvar)  # latent vector

    # --- Generate reconstruction ---
    step = 3  # progressive step: 0=32³, 1=64³, 2=128³, 3=256³
    alpha = 1.0  # fade-in factor (1.0 means fully using current step)
    x_hat = generator(z, step=step, alpha=alpha)

    print("Reconstructed volume shape:", x_hat.shape)

    # --- Discriminator output ---
    d_out = discriminator(x_hat, step=step, alpha=alpha)
    print("Discriminator output shape:", d_out.shape)

