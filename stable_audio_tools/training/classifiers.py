import io

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import typing as tp

from ema_pytorch import EMA
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from safetensors.torch import save_file
from torch.nn import functional as F

from ..models.classifiers import DiTClassifier
from ..inference.sampling import get_alphas_sigmas, truncated_logistic_normal_rescaled, sample_timesteps_logsnr
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_image, log_metric


class DiTClassifierTrainingWrapper(pl.LightningModule):
    '''
    Training wrapper for a DiTClassifier using the ambient diffusion strategy.

    The key idea (from Ambient Omni): low-quality and high-quality audio distributions
    contract under noise and eventually become indistinguishable at high noise levels.
    The classifier is trained across all timesteps t so it can find the per-sample
    "merging point" — the noise level at which a sample can no longer be distinguished
    from the high-quality distribution.

    At each training step:
        1. Sample a noise level t via the configured timestep sampler
        2. Noise the clean latent at that t
        3. Pass (noisy_x, t) through the half-DiT + binary head
        4. Binary cross-entropy loss

    At high t, both label-0 and label-1 samples look identical → classifier outputs ~0.5.
    At low t, the classifier can distinguish them.

    For the good/bad music quality classifier:
        label 0 → low-quality (sPDMX)
        label 1 → high-quality (Jamendo)
    '''

    def __init__(
        self,
        model: DiTClassifier,
        diffusion_objective: str = "rectified_flow",
        pretransform=None,
        pre_encoded: bool = True,
        optimizer_configs: tp.Optional[dict] = None,
        lr: tp.Optional[float] = None,
        label_key: str = "label",
        use_ema: bool = True,
        timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal", "log_snr"] = "uniform",
        timestep_sampler_options: tp.Optional[tp.Dict[str, tp.Any]] = None,
        validation_timesteps: tp.List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    ):
        super().__init__()

        self.model = model
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.pre_encoded = pre_encoded
        self.label_key = label_key

        if use_ema:
            self.model_ema = EMA(
                self.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.model_ema = None

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler
        self.timestep_sampler_options = {} if timestep_sampler_options is None else timestep_sampler_options

        if self.timestep_sampler == "log_snr":
            self.mean_logsnr = self.timestep_sampler_options.get("mean_logsnr", -1.2)
            self.std_logsnr = self.timestep_sampler_options.get("std_logsnr", 2.0)

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "classifier": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {"lr": lr}
                    }
                }
            }
        else:
            if lr is not None:
                print("WARNING: learning_rate and optimizer_configs both specified. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        # Validation
        self.validation_timesteps = validation_timesteps
        self.validation_step_outputs = {}
        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []
            self.validation_step_outputs[f'val/acc_{validation_timestep:.1f}'] = []

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        classifier_opt_config = self.optimizer_configs['classifier']
        opt_classifier = create_optimizer_from_config(classifier_opt_config['optimizer'], self.model.parameters())

        if "scheduler" in classifier_opt_config:
            sched_classifier = create_scheduler_from_config(classifier_opt_config['scheduler'], opt_classifier)
            sched_classifier_config = {
                "scheduler": sched_classifier,
                "interval": "step"
            }
            return [opt_classifier], [sched_classifier_config]

        return [opt_classifier]

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def on_before_zero_grad(self, *args, **kwargs):
        if self.model_ema is not None:
            self.model_ema.update()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, reals):
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
        if self.diffusion_objective in ("rectified_flow", "rf_denoiser"):
            alphas, sigmas = 1 - t, t
        elif self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        else:
            raise ValueError(f"Unknown diffusion_objective: {self.diffusion_objective}")

        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(x)
        return x * alphas + noise * sigmas

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        labels = torch.tensor(
            [md[self.label_key] for md in metadata],
            dtype=torch.long, device=self.device,
        )

        x = self._encode(reals)

        # Sample timesteps
        if self.timestep_sampler == "uniform":
            t = self.rng.draw(x.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(x.shape[0], device=self.device))
        elif self.timestep_sampler == "trunc_logit_normal":
            t = truncated_logistic_normal_rescaled(x.shape[0]).to(self.device)
            t = 1 - t
        elif self.timestep_sampler == "log_snr":
            t = sample_timesteps_logsnr(x.shape[0], mean_logsnr=self.mean_logsnr, std_logsnr=self.std_logsnr).to(self.device)
        else:
            raise ValueError(f"Invalid timestep_sampler: {self.timestep_sampler}")

        noised = self._noise(x, t)
        logits = self.model(noised, t)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        acc = ((logits > 0).long() == labels).float().mean()

        log_dict = {
            "train/loss": loss.detach(),
            "train/acc": acc.detach(),
            "train/std_data": x.std(),
            "train/lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    # ------------------------------------------------------------------
    # Validation — accumulate per-timestep metrics, log at epoch end
    # to mirror DiffusionCondTrainingWrapper's validation pattern
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        labels = torch.tensor(
            [md[self.label_key] for md in metadata],
            dtype=torch.long, device=self.device,
        )

        x = self._encode(reals)

        for validation_timestep in self.validation_timesteps:

            t = torch.full((x.shape[0],), validation_timestep, device=self.device)
            noised = self._noise(x, t)

            with torch.cuda.amp.autocast() and torch.no_grad():
                logits = self.model(noised, t)

                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                acc = ((logits > 0).long() == labels).float().mean()

                self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'].append(loss.item())
                self.validation_step_outputs[f'val/acc_{validation_timestep:.1f}'].append(acc.item())

    def on_validation_epoch_end(self):
        log_dict = {}
        for validation_timestep in self.validation_timesteps:
            loss_key = f'val/loss_{validation_timestep:.1f}'
            acc_key = f'val/acc_{validation_timestep:.1f}'

            val_loss = sum(self.validation_step_outputs[loss_key]) / len(self.validation_step_outputs[loss_key])
            val_acc = sum(self.validation_step_outputs[acc_key]) / len(self.validation_step_outputs[acc_key])

            # Gather losses across all GPUs
            val_loss = self.all_gather(val_loss).mean().item()
            val_acc = self.all_gather(val_acc).mean().item()

            log_metric(self.logger, loss_key, val_loss, step=self.global_step)
            log_metric(self.logger, acc_key, val_acc, step=self.global_step)

        # Get average over all timesteps
        avg_loss = torch.tensor([v for k, vals in self.validation_step_outputs.items() if 'loss' in k for v in vals]).mean()
        avg_acc = torch.tensor([v for k, vals in self.validation_step_outputs.items() if 'acc' in k for v in vals]).mean()

        # Gather across all GPUs
        avg_loss = self.all_gather(avg_loss).mean().item()
        avg_acc = self.all_gather(avg_acc).mean().item()

        log_metric(self.logger, 'val/avg_loss', avg_loss, step=self.global_step)
        log_metric(self.logger, 'val/avg_acc', avg_acc, step=self.global_step)

        # Reset validation outputs
        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []
            self.validation_step_outputs[f'val/acc_{validation_timestep:.1f}'] = []

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, path, use_safetensors=False):
        if self.model_ema is not None:
            model_to_export = self.model_ema.ema_model
        else:
            model_to_export = self.model

        if use_safetensors:
            save_file(model_to_export.state_dict(), path)
        else:
            torch.save({"state_dict": model_to_export.state_dict()}, path)


class DiTClassifierDemoCallback(pl.Callback):
    """
    Demo callback that plots classifier probability vs noise level (t).

    Sweeps t from 0 → 1, noises a cached set of samples from both classes,
    runs the classifier, and plots mean p(label=1 | noisy_x, t) per class.

    The resulting curve directly visualises the merging point — the noise
    level at which the two distributions become indistinguishable and the
    classifier collapses to ~0.5 for both classes.
    """

    def __init__(
        self,
        demo_every: int = 2000,
        num_demos: int = 8,
        t_steps: int = 50,
        label_key: str = "label",
        class_names: tp.Optional[tp.List[str]] = None,
        log_scale_x_axis: bool = False,
        log_scale_stretch: float = 1.00005,
    ):
        """
        Args:
            demo_every: Generate a plot every this many global steps.
            num_demos: Number of samples per class to sweep over.
            t_steps: Number of noise levels to evaluate. Steps are log-spaced
                     via log(a/(a-t)) — the same formula as the x-axis display —
                     so coverage is coarse near t=0 and fine near t=1.
            label_key: Metadata key for the class label (must match the wrapper).
            class_names: Display names for class 0 and class 1.
                         Defaults to ["Low quality (sPDMX)", "High quality (Jamendo)"].
            log_scale_x_axis: If True, plot x-axis as log(a/(a-t)), stretching
                              the high-noise region near t=1. Ticks labelled as t.
            log_scale_stretch: Stretch parameter a > 1 shared by both the t-grid
                               sampling and the x-axis display. Closer to 1 →
                               more extreme concentration near t=1. Default: 1.00005.
        """
        super().__init__()
        self.demo_every = demo_every
        self.num_demos = num_demos
        self.t_steps = t_steps
        self.label_key = label_key
        self.class_names = class_names or ["Low quality (label 0)", "High quality (label 1)"]
        self.log_scale_x_axis = log_scale_x_axis
        self.log_scale_stretch = log_scale_stretch
        self.last_demo_step = -1

        # Cached samples: populated the first time we see a batch with both classes
        self._cached: tp.Optional[tp.Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Cache a representative batch the first time we see one that has
    # samples from both classes
    # ------------------------------------------------------------------

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiTClassifierTrainingWrapper, outputs, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        # Try to fill the cache with num_demos samples from each class
        if self._cached is None:
            labels = torch.tensor([md[self.label_key] for md in metadata], dtype=torch.long)
            idx0 = (labels == 0).nonzero(as_tuple=True)[0]
            idx1 = (labels == 1).nonzero(as_tuple=True)[0]
            if len(idx0) > 0 and len(idx1) > 0:
                self._cached = {
                    "class0": reals[idx0[:self.num_demos]].cpu(),
                    "class1": reals[idx1[:self.num_demos]].cpu(),
                }

        if (trainer.global_step - 1) % self.demo_every != 0 or \
                self.last_demo_step == trainer.global_step or \
                self._cached is None:
            return

        self._generate_plot(trainer, module)

    @rank_zero_only
    @torch.no_grad()
    def _generate_plot(self, trainer, module: DiTClassifierTrainingWrapper):
        self.last_demo_step = trainer.global_step
        module.eval()

        if self.log_scale_x_axis:
            # Log-spaced t-grid via x(t) = log(a/(a-t)), inverse t = a(1-exp(-x)).
            # Uses the same a as the x-axis display so the grid and plot are consistent.
            a = self.log_scale_stretch
            t_end = min(1 - 1e-4, a - 1e-6)
            x_end = float(np.log(a / (a - t_end)))
            t_grid = torch.tensor(
                a * (1 - np.exp(-np.linspace(0, x_end, self.t_steps))),
                dtype=torch.float32, device=module.device,
            )
        else:
            t_grid = torch.linspace(0, 1, self.t_steps, device=module.device)

        # means[cls] and stds[cls] are lists over t_idx
        import numpy as np
        means = {0: [], 1: []}
        stds  = {0: [], 1: []}

        for cls_idx, key in enumerate(["class0", "class1"]):
            x = self._cached[key].to(module.device)
            x = module._encode(x)

            for t_val in t_grid:
                t = torch.full((x.shape[0],), t_val.item(), device=module.device)
                noised = module._noise(x, t)
                logits = module.model(noised, t)
                probs = torch.sigmoid(logits).cpu().numpy()  # (n,)
                means[cls_idx].append(probs.mean())
                stds[cls_idx].append(probs.std())

        # ------------------------------------------------------------------
        # Build the figure
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 4))

        colors = ["#e05c5c", "#5c8fe0"]   # red for bad, blue for good
        t_np = t_grid.cpu().numpy()

        if self.log_scale_x_axis:
            a = self.log_scale_stretch  # singularity at t=a; a>1 keeps [0,1] finite
            # x(t) = log(a / (a - t)),  inverse: t = a * (1 - exp(-x))
            x_vals = np.log(a / (a - t_np))
        else:
            x_vals = t_np

        mean0 = np.array(means[0])
        mean1 = np.array(means[1])
        std0  = np.array(stds[0])
        std1  = np.array(stds[1])

        # ------------------------------------------------------------------
        # Compute marker locations (interpolated)
        # A) first t where high-quality (class 1) curve drops below 0.5
        # B) where high-quality and low-quality curves cross
        # ------------------------------------------------------------------
        marker_style = dict(color="gray", linestyle="--", linewidth=1, alpha=0.7)

        t_below50 = x_below50 = None
        for i in range(len(mean1)):
            if not np.isnan(mean1[i]) and mean1[i] < 0.5:
                if i == 0:
                    t_below50, x_below50 = t_np[0], x_vals[0]
                else:
                    frac = (0.5 - mean1[i-1]) / (mean1[i] - mean1[i-1])
                    t_below50 = float(t_np[i-1] + frac * (t_np[i] - t_np[i-1]))
                    x_below50 = float(x_vals[i-1] + frac * (x_vals[i] - x_vals[i-1]))
                break

        t_cross = x_cross = y_cross = None
        diff = mean1 - mean0
        for i in range(1, len(diff)):
            if not (np.isnan(diff[i-1]) or np.isnan(diff[i])) and diff[i-1] > 0 >= diff[i]:
                frac = diff[i-1] / (diff[i-1] - diff[i])
                t_cross = float(t_np[i-1] + frac * (t_np[i] - t_np[i-1]))
                x_cross = float(x_vals[i-1] + frac * (x_vals[i] - x_vals[i-1]))
                y_cross = float(mean1[i-1] + frac * (mean1[i] - mean1[i-1]))
                break

        print(f"[classifier demo step {trainer.global_step}]")
        print(f"  High quality first drops below 0.5:  t = {t_below50:.6f}" if t_below50 is not None else "  High quality never drops below 0.5")
        print(f"  High quality / low quality crossing:  t = {t_cross:.6f},  p = {y_cross:.4f}" if t_cross is not None else "  Curves do not cross")

        for cls_idx, (mean, std) in enumerate([(mean0, std0), (mean1, std1)]):
            ax.plot(x_vals, mean, color=colors[cls_idx], linewidth=2, label=self.class_names[cls_idx])
            ax.fill_between(
                x_vals,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=colors[cls_idx],
                alpha=0.2,
            )

        # Decision boundary — no legend entry
        ax.axhline(0.5, **marker_style)

        # Vertical + horizontal markers
        if x_below50 is not None:
            ax.axvline(x_below50, **marker_style)
        if x_cross is not None:
            ax.axvline(x_cross, **marker_style)
            ax.axhline(y_cross, **marker_style)

        ax.set_xlabel("Noise level (t)")
        ax.set_ylabel("p(high quality | noisy x, t)")
        ax.set_title(f"Classifier probability vs noise level — step {trainer.global_step}")
        ax.set_xlim(x_vals[0], x_vals[-1])
        ax.set_ylim(0, 1)

        if self.log_scale_x_axis:
            # Ticks evenly spaced in log(a/(a-t)) space, labelled as t values
            # inverse: t = a * (1 - exp(-x))
            tick_x = np.linspace(x_vals[0], x_vals[-1], 9)
            tick_t = a * (1 - np.exp(-tick_x))
            ax.set_xticks(tick_x)
            ax.set_xticklabels([f"{v:.2f}" for v in tick_t])

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Render to a PIL image and log
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # force eager decode before buf can be GC'd
        plt.close(fig)

        log_image(trainer.logger, "demo_classifier_probs_vs_t", img)

        module.train()
