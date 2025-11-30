# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # ---------------------------
# # Helper: freeze everything
# # ---------------------------
# def freeze_all(model):
#     for p in model.parameters():
#         p.requires_grad = False
#
#
# # ---------------------------
# # Helper: unfreeze levels (base / level_64 / level_128 / level_256)
# # ---------------------------
# def enable_levels(model, levels: set):
#     """
#     Unfreezes:
#         - base parts: fc / base_conv / to_voxel / final_conv
#         - progressive blocks in model.levels dict
#     """
#     for p in model.parameters():
#         p.requires_grad = False
#
#     # base
#     if "base" in levels:
#         for name in ["fc", "base_conv", "to_voxel", "final_conv"]:
#             if hasattr(model, name):
#                 for p in getattr(model, name).parameters():
#                     p.requires_grad = True
#
#     # progressive levels
#     if hasattr(model, "levels"):
#         for lvl in levels:
#             if lvl in model.levels:
#                 for p in model.levels[lvl].parameters():
#                     p.requires_grad = True
#
#
# # ---------------------------
# # Get trainable params from model
# # ---------------------------
# def params_of(model):
#     return [p for p in model.parameters() if p.requires_grad]
#
#
# # ---------------------------
# # Dummy dataset
# # ---------------------------
# class ToyDataset:
#     def __init__(self, img_size=64, voxel_res=32):
#         self.img_size = img_size
#         self.voxel_res = voxel_res
#
#     def get_batch(self, B):
#         x_img = torch.randn(B, 1, self.img_size, self.img_size, device=device)
#         x_vox = (torch.rand(B, 1, self.voxel_res, self.voxel_res, self.voxel_res, device=device) > 0.5).float()
#         return x_img, x_vox
#
#
# # ---------------------------
# # Progressive schedule
# # ---------------------------
# STAGES = [
#     {"res": 32,  "phase": "stabilize", "steps": 1000},
#     {"res": 64,  "phase": "fade_in",   "steps": 1000},
#     {"res": 64,  "phase": "stabilize", "steps": 1000},
#     {"res": 128, "phase": "fade_in",   "steps": 1500},
#     {"res": 128, "phase": "stabilize", "steps": 1500},
#     {"res": 256, "phase": "fade_in",   "steps": 2000},
#     {"res": 256, "phase": "stabilize", "steps": 2000},
# ]
#
#
# # ---------------------------
# # Models (reuse your definitions)
# # ---------------------------
# G = ProgressiveGenerator(n_latent=50).to(device)
# D = ProgressiveDiscriminator().to(device)
# E = Encoder2D(n_latent=50).to(device)
#
# bce = nn.BCELoss()
# mse = nn.MSELoss()
#
# dataset = ToyDataset()
#
# # ---------------------------
# # Training main loop
# # ---------------------------
# global_step = 0
#
# for stage in STAGES:
#     res    = stage["res"]
#     phase  = stage["phase"]
#     steps  = stage["steps"]
#
#     print(f"\n=== Stage {res} - {phase} ({steps} steps) ===")
#
#     # which levels should be active in this resolution?
#     levels = {"base"}
#     if res >= 64:  levels.add("level_64")
#     if res >= 128: levels.add("level_128")
#     if res >= 256: levels.add("level_256")
#
#     # encoder always trains
#     for p in E.parameters():
#         p.requires_grad = True
#
#     # ---- freeze everything first ----
#     freeze_all(G)
#     freeze_all(D)
#
#     # progressive rules
#     if phase == "fade_in":
#         # only new level + base
#         if res == 64:  new_lvl = "level_64"
#         if res == 128: new_lvl = "level_128"
#         if res == 256: new_lvl = "level_256"
#
#         enable_levels(G, {"base", new_lvl})
#         enable_levels(D, {"base", new_lvl})
#
#     else:  # stabilize
#         enable_levels(G, levels)
#         enable_levels(D, levels)
#
#     # build optimizers per stage
#     G_opt = optim.RMSprop(params_of(G), lr=1e-4)
#     D_opt = optim.RMSprop(params_of(D), lr=5e-5)
#     E_opt = optim.Adam(params_of(E),  lr=1e-4)
#
#     # ---------------------------
#     # Stage training loop
#     # ---------------------------
#     for step in range(steps):
#
#         # alpha for fade-in
#         alpha = (step + 1) / steps if phase == "fade_in" else 1.0
#
#         # get batch
#         x_img, x_vox_base = dataset.get_batch(B=8)
#
#         # upsample GT voxel to target res
#         if res == 32:
#             x_vox = x_vox_base
#         else:
#             x_vox = F.interpolate(x_vox_base, size=(res, res, res), mode="nearest")
#
#         # ---------------------------
#         # Encode → latent
#         # ---------------------------
#         z, mu, logvar = E(x_img)
#
#         # ---------------------------
#         # Generate
#         # ---------------------------
#         fake = G(z, target_res=res, alpha=alpha)
#
#         # ---------------------------
#         # D forward
#         # ---------------------------
#         D_real = D(x_vox,  input_res=res, alpha=alpha)
#         D_fake = D(fake.detach(), input_res=res, alpha=alpha)
#
#         real_y = torch.ones_like(D_real)
#         fake_y = torch.zeros_like(D_fake)
#
#         # ---------------------------
#         # Discriminator loss
#         # ---------------------------
#         D_loss = bce(D_real, real_y) + bce(D_fake, fake_y)
#
#         D_opt.zero_grad()
#         D_loss.backward()
#         D_opt.step()
#
#         # ---------------------------
#         # Generator + Encoder loss
#         # ---------------------------
#         D_fake_G = D(fake, input_res=res, alpha=alpha)
#         G_adv = bce(D_fake_G, real_y)
#
#         recon = mse(fake, x_vox)
#         KL = -0.5 * torch.mean(1 + 2 * logvar - mu.pow(2) - torch.exp(2 * logvar))
#
#         alpha1 = 5.0
#         alpha2 = 5e-4
#
#         VAE_loss = alpha1 * KL + alpha2 * recon
#         G_loss = G_adv + VAE_loss
#
#         G_opt.zero_grad()
#         E_opt.zero_grad()
#         G_loss.backward()
#         G_opt.step()
#         E_opt.step()
#
#         if step % 100 == 0:
#             print(f"[{step:05d}/{steps}] α={alpha:.3f} | "
#                   f"D={D_loss.item():.4f} | G={G_loss.item():.4f} | "
#                   f"Recon={recon.item():.4f} | KL={KL.item():.4f}")
#
#         global_step += 1
#
#     torch.save({
#         "G": G.state_dict(),
#         "D": D.state_dict(),
#         "E": E.state_dict(),
#         "global_step": global_step
#     }, f"chkpt_{res}_{phase}.pth")
#
# print("Training done!")
