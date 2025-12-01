import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from monai.metrics import DiceMetric


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
        self.recon_target = getattr(cfg, "recon_target", "mask")
        self.use_epoch_phase = getattr(cfg, "use_epoch_phase", False)
        self.max_epochs_per_step = getattr(cfg, "max_epochs_per_step", None)

        # progressive state
        self.current_step = getattr(cfg, "initial_step", 0)
        self.alpha = 1.0 if self.current_step == 0 else 0.
        self._phase_iter_counter = 0
        self._alpha_ramp_counter = 0
        self._global_iter = 0

        # losses
        if self.recon_loss_type == "bce":
            self.recon_loss_fn = lambda out, tgt: F.binary_cross_entropy(out, tgt)
        elif self.recon_loss_type == "l1":
            self.recon_loss_fn = lambda out, tgt: F.l1_loss(out, tgt)
        else:
            raise ValueError("recon_loss must be 'bce' or 'l1'")

        self._apply_freezing()
        self.val_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

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

    def resize_on_step(self, image):
        res = self.cfg.init_res * 2 ** self.current_step
        output = F.interpolate(image, (res, res, res), mode='trilinear')
        return output
    # ============================================================
    # TRAINING STEP
    # ============================================================
    def training_step(self, batch, batch_idx):
        image_synth, image_oof, mask, mask_oof = self._get_input_and_target(batch)
        image_synth = self.resize_on_step(image_synth)
        x_hat, mu, logvar, z = self.forward(image_synth)

        recon = self.recon_loss_fn(x_hat, image_synth)
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
        x_hat, mu, logvar, z = self.forward(image_oof)
        pred_bin = x_hat > self.cfg.threshold
        self.val_dice_metric(y_pred=pred_bin, y=(mask * (mask != 2)).long())

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
        self.log("val_dice", dice_score, prog_bar=True, rank_zero_only=True)
        self.val_dice_metric.reset()
        if self.use_epoch_phase and (self.max_epochs_per_step is not None):
            desired_step = min(self.max_steps, self.current_epoch // max(1, self.max_epochs_per_step))
            if desired_step != self.current_step:
                self.current_step = desired_step
                self.alpha = 1.0
                self._apply_freezing()
                print(f"[Progressive] Epoch step update → {self.current_step}")