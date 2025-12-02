import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from monai.metrics import DiceMetric
from losses import *


class ProgressiveVAETrainer(pl.LightningModule):
    """
    Trainer with **progressive growing + automatic parameter freezing**.
    """
    def __init__(self, encoder: nn.Module, generator: nn.Module, cfg):
        super().__init__()
        self.save_hyperparameters(dict(
            cfg=vars(cfg) if hasattr(cfg, '__dict__') else cfg
        ))

        self.encoder = encoder
        self.generator = generator

        # cfg defaults
        self.cfg = cfg
        self.max_steps = getattr(cfg, "max_steps", 3)
        self.lr = getattr(cfg, "lr", 1e-4)
        self.beta_kl = getattr(cfg, "beta_kl", 1.0)
        self.recon_loss_type = getattr(cfg, "recon_loss", "bce")
        self.alpha_ramp_iters = getattr(cfg, "alpha_ramp_iters", 2000)
        self.phase_iters = getattr(cfg, "phase_iters", 2000)
        self.use_epoch_phase = getattr(cfg, "use_epoch_phase", False)
        self.max_epochs_per_step = getattr(cfg, "max_epochs_per_step", None)

        # progressive state
        self.current_step = getattr(cfg, "initial_step", 0)
        self.alpha = 1.0 if self.current_step == 0 else 0.
        self._phase_iter_counter = 0
        self._alpha_ramp_counter = 0
        self._global_iter = 0
        self._apply_freezing()
        self.val_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
        self.val_topo_losses = []
        self.val_surf_losses = []
        self.val_num_samples = 0
        self.topo_loss = FastClDiceLoss()  # ← main topology driver
        self.surf_loss = SurfaceLoss(tau_vox=2.0)  # ← main SurfaceDice driver

    def freeze_progressive_layers(self, model, current_step):
        """
        Freeze all blocks < current_step for any ProgressiveEncoder3D or ProgressiveGenerator3D.
        Works based on index parsing: blocks.0, blocks.1, ...
        """
        for name, module in model.named_modules():
            # Layers with step index
            if any(key in name for key in ["blocks", "to_voxel", "to_latent_list", "from_prev_blocks", "from_voxel"]):
                parts = name.split(".")
                if len(parts) >= 2 and parts[-1].isdigit():
                    idx = int(parts[-1])

                    if idx < current_step:
                        for p in module.parameters():
                            p.requires_grad = False
                        module.eval()  # freeze BatchNorm
                    else:
                        for p in module.parameters():
                            p.requires_grad = True
                        module.train()

    def _apply_freezing(self):
        """Call this whenever current_step is updated."""
        self.freeze_progressive_layers(self.encoder, self.current_step)
        self.freeze_progressive_layers(self.generator, self.current_step)

    # ============================================================
    # FORWARD
    # ============================================================
    def forward(self, x, step=None, alpha=None):
        if step is None: step = self.current_step
        if alpha is None: alpha = self.alpha

        mu, logvar = self.encoder(x, step=step, alpha=alpha)
        z = self.encoder.reparameterize(mu, logvar)
        x_hat = self.generator(z, step=step, alpha=alpha)
        return x_hat, mu, logvar, z

    # ============================================================
    # DATA HANDLING
    # ============================================================
    def _get_input_and_target(self, batch):
        image = batch.get('Image')
        mask = batch.get('Mask')
        mask_oof = batch['Mask_OOF']
        image_synth = image * mask * (mask != 2) + image * mask_oof * (mask == 2)
        image_oof = image * mask_oof
        return image_synth, image_oof, mask, mask_oof

    def resize_on_step(self, image, res=None):
        if res is None:
            res = self.cfg.init_res * 2 ** self.current_step
        if type(res) == tuple:
            output = F.interpolate(image, res, mode='trilinear')
        else:
            output = F.interpolate(image, (res, res, res), mode='trilinear')
        return output
    # ============================================================
    # TRAINING STEP
    # ============================================================
    def training_step(self, batch, batch_idx):
        image_synth, image_oof, mask, mask_oof = self._get_input_and_target(batch)
        image_synth = self.resize_on_step(image_synth)
        x_hat, mu, logvar, z = self.forward(image_synth)

        recon = confidence_weighted_l1(x_hat, image_synth, mask, self.cfg.pseudo_weight)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon + self.beta_kl * kl

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/alpha", torch.tensor(self.alpha, device=self.device), on_step=True)

        # update progressive schedule
        prev_step = self.current_step
        self._update_progressive_state()
        self._global_iter += 1

        if self.current_step != prev_step:
            self._apply_freezing()
            print(f"[Progressive] Step changed: {prev_step} → {self.current_step}. Freezing applied.")
        return loss

    # ============================================================
    # VALIDATION
    # ============================================================
    def validation_step(self, batch, batch_idx):
        image_synth, image_oof, mask, mask_oof = self._get_input_and_target(batch)
        image_oof = self.resize_on_step(image_oof)
        x_hat, mu, logvar, z = self.forward(image_oof)
        x_hat = self.resize_on_step(x_hat, tuple(mask.shape[2:]))
        pred_bin = x_hat > 0
        topo_loss = self.topo_loss(pred_bin, mask* (mask != 2))
        surf_loss = self.surf_loss(pred_bin, mask* (mask != 2))
        self.val_dice_metric(y_pred=pred_bin, y=(mask * (mask != 2)).long())
        batch_size = image_synth.shape[0]
        self.val_topo_losses.append(topo_loss * batch_size)
        self.val_surf_losses.append(surf_loss * batch_size)
        self.val_num_samples += batch_size

    # ============================================================
    # OPTIMIZER
    # ============================================================
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.generator.parameters()),
            lr=self.lr, weight_decay=1e-2
        )
        return opt

    # ============================================================
    # PROGRESSIVE SCHEDULING (UNMODIFIED)
    # ============================================================
    def _update_progressive_state(self):
        if self.max_steps <= 0:
            return

        if self.use_epoch_phase and (self.max_epochs_per_step is not None):
            self.alpha = 1.0
            return

        # ramp alpha from 0→1
        if self.alpha < 1.0:
            self._alpha_ramp_counter += 1
            self.alpha = min(1.0, self._alpha_ramp_counter / max(1, self.alpha_ramp_iters))
            if self.alpha >= 1.0:
                self._phase_iter_counter = 0
        else:
            # alpha full, count phase iterations
            self._phase_iter_counter += 1
            if self._phase_iter_counter >= max(1, self.phase_iters):
                if self.current_step < self.max_steps:
                    self.current_step += 1
                    self._alpha_ramp_counter = 0
                    self.alpha = 0.0 if self.current_step > 0 else 1.0
                    self._phase_iter_counter = 0

    def on_validation_epoch_end(self):
        dice_score = self.val_dice_metric.aggregate().mean().item()
        avg_topo_loss = torch.stack(self.val_topo_losses).sum() / self.val_num_samples
        avg_surf_loss = torch.stack(self.val_surf_losses).sum() / self.val_num_samples
        topo_score = 1.0 - avg_topo_loss
        surf_score = 1.0 - avg_surf_loss
        comp_metric = 0.30 * topo_score + 0.35 * surf_score + 0.35 * dice_score
        # === Logging ===
        self.log("val_dice", dice_score, prog_bar=True, rank_zero_only=True)
        self.log("val_topo_score", topo_score, prog_bar=True, rank_zero_only=True)
        self.log("val_surf_score", surf_score, prog_bar=True, rank_zero_only=True)
        self.log("val_comp_metric", comp_metric, prog_bar=True, rank_zero_only=True, sync_dist=True)

        if self.trainer.is_global_zero:
            print(f"\nVAL Epoch {self.current_epoch:03d} │ "
                  f"Dice: {dice_score:.4f} │ "
                  f"Topo: {topo_score:.4f} │ "
                  f"Surf: {surf_score:.4f} │ "
                  f"→ COMP: {comp_metric:.4f} ←\n")

        # === Reset everything ===
        self.val_dice_metric.reset()
        self.val_topo_losses.clear()
        self.val_surf_losses.clear()
        self.val_num_samples = 0

        if self.use_epoch_phase and (self.max_epochs_per_step is not None):
            desired_step = min(self.max_steps, self.current_epoch // max(1, self.max_epochs_per_step))
            if desired_step != self.current_step:
                self.current_step = desired_step
                self.alpha = 1.0
                self._apply_freezing()
                print(f"[Progressive] Epoch step update → {self.current_step}")