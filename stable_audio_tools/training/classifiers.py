import pytorch_lightning as pl
import torch
import typing as tp

from torch.nn import functional as F

from ..models.classifiers import DiTClassifier
from ..inference.sampling import get_alphas_sigmas
from .utils import create_optimizer_from_config, create_scheduler_from_config

# Mirrors the ambient diffusion constants used in diffusion.py
DEFAULT_MIN_NOISE_LEVEL = 0.0
MIN_NOISE_LEVEL, MAX_NOISE_LEVEL = 0.0, 1.0


class AmbientDiTClassifierTrainingWrapper(pl.LightningModule):
    """
    Training wrapper for a DiTClassifier using the ambient diffusion setup.

    Each training sample is noised to a randomly sampled timestep t drawn
    from [min_noise_level, 1.0] (per-sample min_noise_level comes from
    batch metadata, mirroring the ambient diffusion training).  The noisy
    latent is passed to the classifier, and the loss is cross-entropy against
    the integer label stored in metadata under `label_key`.

    For the good/bad music quality classifier:
        label 0 → low-quality (sPDMX)
        label 1 → high-quality (Jamendo)
    """

    def __init__(
        self,
        model: DiTClassifier,
        diffusion_objective: str = "rectified_flow",
        pretransform=None,
        pre_encoded: bool = True,
        optimizer_configs: tp.Optional[dict] = None,
        lr: tp.Optional[float] = None,
        label_key: str = "label",
    ):
        """
        Args:
            model: Instantiated DiTClassifier.
            diffusion_objective: Noise schedule type — "rectified_flow" or "v".
            pretransform: Optional VAE pretransform for encoding raw audio.
                          Only needed when pre_encoded=False.
            pre_encoded: If True, batch reals are already VAE latents.
            optimizer_configs: Dict matching the training.optimizer_configs
                               schema, keyed by "classifier".
            lr: Simple learning rate shorthand — used if optimizer_configs
                is not provided.
            label_key: Metadata key that holds the integer class label.
        """
        super().__init__()

        self.model = model
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.pre_encoded = pre_encoded
        self.label_key = label_key

        # Quasi-random timestep sampler (same as diffusion trainers)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        assert lr is not None or optimizer_configs is not None, \
            "Specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "classifier": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {"lr": lr}
                    }
                }
            }

        self.optimizer_configs = optimizer_configs

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cls_opt_cfg = self.optimizer_configs["classifier"]
        opt = create_optimizer_from_config(cls_opt_cfg["optimizer"], self.model.parameters())

        if "scheduler" in cls_opt_cfg:
            sched = create_scheduler_from_config(cls_opt_cfg["scheduler"], opt)
            return [opt], [{"scheduler": sched, "interval": "step"}]

        return [opt]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _encode(self, reals):
        """Apply pretransform encoding + scale correction if needed."""
        x = reals
        if self.pretransform is not None:
            self.pretransform.to(self.device)
            if not self.pre_encoded:
                with torch.set_grad_enabled(self.pretransform.enable_grad):
                    x = self.pretransform.encode(x)
            else:
                if hasattr(self.pretransform, "scale") and self.pretransform.scale != 1.0:
                    x = x / self.pretransform.scale
        return x

    def _noise(self, x, t):
        """Add noise at timestep t using the configured diffusion objective."""
        if self.diffusion_objective in ("rectified_flow", "rf_denoiser"):
            t_3d = t[:, None, None]
            alphas, sigmas = 1.0 - t_3d, t_3d
        elif self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
            alphas, sigmas = alphas[:, None, None], sigmas[:, None, None]
        else:
            raise ValueError(f"Unknown diffusion_objective: {self.diffusion_objective}")

        noise = torch.randn_like(x)
        return x * alphas + noise * sigmas

    def _sample_ambient_t(self, batch_size, metadata):
        """Sample t in [min_noise_level, 1.0] per sample (ambient-style).

        min_noise_level defaults to 0.0 if absent from metadata, which is
        correct for a classifier — we want to train across all noise levels.
        """
        t = self.rng.draw(batch_size)[:, 0].to(self.device)

        min_levels = torch.tensor(
            [md.get("min_noise_level", DEFAULT_MIN_NOISE_LEVEL) for md in metadata],
            dtype=t.dtype, device=self.device,
        ).clamp(MIN_NOISE_LEVEL, MAX_NOISE_LEVEL)

        return min_levels + t * (MAX_NOISE_LEVEL - min_levels)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        labels = torch.tensor(
            [md[self.label_key] for md in metadata],
            dtype=torch.long, device=self.device
        )

        x = self._encode(reals)
        t = self._sample_ambient_t(x.shape[0], metadata)
        noised = self._noise(x, t)

        logits = self.model(noised, t)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    # ------------------------------------------------------------------
    # Validation — evaluate accuracy at several fixed noise levels
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        labels = torch.tensor(
            [md[self.label_key] for md in metadata],
            dtype=torch.long, device=self.device
        )

        x = self._encode(reals)

        val_losses, val_accs = [], []
        for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
            t = torch.full((x.shape[0],), t_val, dtype=x.dtype, device=self.device)
            noised = self._noise(x, t)

            with torch.no_grad():
                logits = self.model(noised, t)

            loss = F.cross_entropy(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            val_losses.append(loss)
            val_accs.append(acc)

            self.log(f"val/loss_t{t_val:.1f}", loss, sync_dist=True)
            self.log(f"val/acc_t{t_val:.1f}", acc, sync_dist=True)

        avg_loss = torch.stack(val_losses).mean()
        avg_acc = torch.stack(val_accs).mean()
        self.log("val/loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", avg_acc, prog_bar=True, sync_dist=True)
        return avg_loss
