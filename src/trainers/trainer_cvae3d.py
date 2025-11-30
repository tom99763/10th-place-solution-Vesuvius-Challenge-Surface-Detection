import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch


class ProgressiveVAETrainer(pl.LightningModule):
    """
    Trainer for a progressive-growing 3D VAE (encoder + generator).
    Expects:
      cfg.max_steps (int)             : number of progressive steps (matches models)
      cfg.lr (float)                  : learning rate
      cfg.beta_kl (float)             : KL weight (beta-VAE)
      cfg.recon_loss (str)            : 'bce' or 'l1' (default 'bce')
      cfg.alpha_ramp_iters (int)      : iterations to linearly ramp alpha from 0->1 within a phase
      cfg.phase_iters (int)           : additional iterations to keep alpha==1 before increasing step
      cfg.max_epochs_per_step (int)   : optional alternative to phase_iters (one may choose epochs)
      cfg.recon_target (str)          : 'mask' or 'image' (default 'mask' if available)
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
        self.recon_target = getattr(cfg, "recon_target", "mask")  # 'mask' prefered
        self.use_epoch_phase = getattr(cfg, "use_epoch_phase", False)  # optional alternate scheduling
        self.max_epochs_per_step = getattr(cfg, "max_epochs_per_step", None)

        # internal progressive state
        self.current_step = 0
        self.alpha = 1.0 if self.current_step == 0 else 0.0
        self._phase_iter_counter = 0
        self._alpha_ramp_counter = 0
        self._global_iter = 0

        # losses
        if self.recon_loss_type == "bce":
            # generator returns sigmoid(prob); use binary CE
            self.recon_loss_fn = lambda out, tgt: F.binary_cross_entropy(out, tgt)
        elif self.recon_loss_type == "l1":
            self.recon_loss_fn = lambda out, tgt: F.l1_loss(out, tgt)
        else:
            raise ValueError("recon_loss must be 'bce' or 'l1'")

    def forward(self, x, step=None, alpha=None):
        """Encode -> Reparameterize -> Decode"""
        if step is None:
            step = self.current_step
        if alpha is None:
            alpha = self.alpha
        mu, logvar = self.encoder(x, step=step, alpha=alpha)
        z = self.encoder.reparameterize(mu, logvar)
        x_hat = self.generator(z, step=step, alpha=alpha)
        return x_hat, mu, logvar, z

    def _get_input_and_target(self, batch):
        """
        Prefer reconstructing mask (segmentation) if present and recon_target=='mask'.
        Otherwise reconstruct image.
        Ensure shapes and dtypes match generator output.
        """
        image = batch.get('Image') if isinstance(batch, dict) else batch[0]
        mask = batch.get('Mask') if isinstance(batch, dict) else None
        # choose target
        if self.recon_target == "mask" and mask is not None:
            target = (mask > 0).float()  # Make it probabilistic / binary
        else:
            target = image.float()
        # input to encoder: try Image first, else Mask_OOF or Mask
        if 'Encoder_Input' in batch:
            inp = batch['Encoder_Input'].float()
        elif 'Mask_OOF' in batch:
            # if you previously concatenated image+mask_oof in diffeo trainer, change here.
            inp = batch['Mask_OOF'].float()
        else:
            inp = image.float()
        # Ensure single channel ordering if needed
        return inp, target

    def training_step(self, batch, batch_idx):
        inp, target = self._get_input_and_target(batch)
        # encoder/generator expect shapes: [B, C, D, H, W]
        # Ensure target is same shape as generator output channels
        x_hat, mu, logvar, z = self.forward(inp, step=self.current_step, alpha=self.alpha)

        # recon loss (generator outputs sigmoid probabilities)
        recon = self.recon_loss_fn(x_hat, target)

        # KL divergence (mean over batch)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = recon + self.beta_kl * kl

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon", recon, on_step=True, on_epoch=True)
        self.log("train/kl", kl, on_step=True, on_epoch=True)
        self.log("train/step", torch.tensor(self.current_step, device=self.device).float(), on_step=True, on_epoch=False)
        self.log("train/alpha", torch.tensor(self.alpha, device=self.device).float(), on_step=True, on_epoch=False)

        # update progressive scheduling counters (performed every training iteration)
        self._update_progressive_state()

        self._global_iter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        inp, target = self._get_input_and_target(batch)
        x_hat, mu, logvar, z = self.forward(inp, step=self.current_step, alpha=self.alpha)

        recon = self.recon_loss_fn(x_hat, target)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon + self.beta_kl * kl

        # Optional metrics: compute Dice between binarized x_hat and target if target is binary
        dice = None
        try:
            if target.dtype.is_floating_point:
                targ_bin = (target > 0.5).long()
                pred_bin = (x_hat > 0.5).long()
                inter = (pred_bin & targ_bin).float().sum()
                union = pred_bin.float().sum() + targ_bin.float().sum()
                dice = 2.0 * inter / (union + 1e-6)
                dice = dice.item()
                self.log("val/dice", dice, on_epoch=True, prog_bar=True, rank_zero_only=True)
        except Exception:
            pass

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, rank_zero_only=True)
        self.log("val/recon", recon, on_epoch=True, prog_bar=False)
        self.log("val/kl", kl, on_epoch=True, prog_bar=False)

        # optionally return a small visualization batch (first sample) for callbacks
        if batch_idx == 0:
            return dict(
                input=inp.detach().cpu(),
                recon=x_hat.detach().cpu(),
                target=target.detach().cpu()
            )
        return None

    def configure_optimizers(self):
        # Single optimizer for encoder+generator
        opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.generator.parameters()),
            lr=self.lr,
            weight_decay=1e-2
        )
        return opt

    # -----------------------------
    # Progressive scheduling utils
    # -----------------------------
    def _update_progressive_state(self):
        """
        Called every training_step to update alpha and possibly bump current_step.
        Behavior:
          - Ramp alpha from 0 -> 1 across self.alpha_ramp_iters iters.
          - Once alpha reaches 1.0, increment a phase counter. After phase_iters iters with alpha==1,
            step is increased by 1 (until max_steps), then alpha reset to 0 for next stage.
        Alternative epoch-based scheduling available via self.max_epochs_per_step if set.
        """
        # If only one step, nothing to do
        if self.max_steps <= 0:
            return

        # If epoch-based mode is requested and provided, we don't change alpha per iteration here.
        if self.use_epoch_phase and (self.max_epochs_per_step is not None):
            # alpha remains 1.0 during a step; step advancement handled in on_epoch_end
            self.alpha = 1.0
            return

        # Ramp alpha up
        if self.alpha < 1.0:
            self._alpha_ramp_counter += 1
            if self.alpha_ramp_iters <= 0:
                self.alpha = 1.0
            else:
                self.alpha = float(self._alpha_ramp_counter) / float(max(1, self.alpha_ramp_iters))
                self.alpha = min(1.0, self.alpha)

            # if alpha reached 1, start phase counter
            if self.alpha >= 1.0:
                self._phase_iter_counter = 0
        else:
            # alpha == 1.0: increment phase counter and possibly move to next step
            self._phase_iter_counter += 1
            if self._phase_iter_counter >= max(1, self.phase_iters):
                # increment step if possible
                if self.current_step < self.max_steps:
                    self.current_step += 1
                    # reset alpha ramp counters for next stage
                    self._alpha_ramp_counter = 0
                    self.alpha = 0.0 if self.current_step > 0 else 1.0
                    self._phase_iter_counter = 0
                    # log change
                    self.log("train/current_step", torch.tensor(self.current_step, device=self.device).float(), on_step=True, on_epoch=False)
                else:
                    # already at max, keep alpha=1
                    self.alpha = 1.0

    # Optional epoch-end hook for epoch-based step scheduling
    def on_validation_epoch_end(self):
        if self.use_epoch_phase and (self.max_epochs_per_step is not None):
            # increment step after completing max_epochs_per_step
            epoch = self.current_epoch
            desired_step = min(self.max_steps, epoch // max(1, self.max_epochs_per_step))
            if desired_step != self.current_step:
                self.current_step = desired_step
                self.alpha = 1.0
                self.log("train/current_step", torch.tensor(self.current_step, device=self.device).float(), on_epoch=True)
