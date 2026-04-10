import typing as tp
from typing import Optional
import torch
import math

from torch.nn import functional as F
from torch import nn
from .utils import mmd

import ambient_utils

from ...inference.sampling import get_alphas_sigmas

class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0, decay = 1.0):
        super().__init__()

        self.name = name
        self.decay = float(decay)
        self.master_weight = weight
        weight = torch.tensor(float(weight))
        self.register_buffer('weight', weight)

    def decay_weight(self):
        if self.decay != 1.0:
            self.weight *= self.decay
        elif self.decay == 1.0 and self.weight != self.master_weight:
            self.weight = torch.tensor(self.master_weight, dtype=self.weight.dtype, device=self.weight.device)
    def forward(self, info, *args, **kwargs):
        raise NotImplementedError

class ValueLoss(LossModule):
    def __init__(self, key: str, name, weight: float = 1.0, decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key = key

    def forward(self, info):
        self.decay_weight()
        return self.weight * info[self.key]

class TargetValueLoss(LossModule):
    def __init__(self, key: str, target: float, name: str, weight: float = 1.0):
        super().__init__(name=name, weight=weight)

        self.key = key
        self.target = target
    
    def forward(self, info):
        self.decay_weight()
        return self.weight * (info[self.key] - self.target).abs()

class L1Loss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'l1_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        l1_loss = F.l1_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info:
            l1_loss = l1_loss[info[self.mask_key]]

        l1_loss = l1_loss.mean()
        self.decay_weight()
        return self.weight * l1_loss

class MSELoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'mse_loss',decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        mse_loss = F.mse_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]

            if mask.ndim == 2 and mse_loss.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != mse_loss.shape[1]:
                mask = mask.repeat(1, mse_loss.shape[1], 1)

            mse_loss = mse_loss[mask]

        mse_loss = mse_loss.mean()
        self.decay_weight()
        return self.weight * mse_loss


class AmbientOLoss(LossModule):
    def __init__(
        self,
        diffusion_objective: str,
        name: str = 'ambient_loss',
        weight: float = 1.0,
        decay = 1.0,
        use_ambient_weight: bool = False,
        buffer_threshold: float = 2.0 / math.sqrt(3),
        clip_max_weight: Optional[float] = None,
    ):
        super().__init__(name=name, weight=weight, decay=decay)

        self.diffusion_objective = diffusion_objective
        self.use_ambient_weight = use_ambient_weight
        self.buffer_threshold = buffer_threshold
        self.clip_max_weight = clip_max_weight

    def compute_ambient_weight(self, sigma_t, sigma_t_min):
        # Implements λ_amb(σ, σ_min) = σ^4 / (σ^2 - σ^2_min)^2
        numerator = sigma_t**4
        denominator = (sigma_t**2 - sigma_t_min**2) ** 2

        # Add epsilon to avoid division by zero
        weight = numerator / (denominator + 1e-10)

        # Optional: clip to maximum value
        if self.clip_max_weight is not None:
            weight = torch.clamp(weight, max=self.clip_max_weight)

        return weight

    def apply_buffer(self, sigma_t, sigma_t_min):
        # Buffer condition: σ_t > threshold * σ_{t_min}
        # For threshold = 2/√3 ≈ 1.15, this means σ_t must be at least 15% larger
        buffer_mask = sigma_t > (self.buffer_threshold * sigma_t_min)

        # Safety: ensure at least some samples pass the buffer
        # (in case all samples are in buffer zone due to bad sampling)
        if buffer_mask.sum() == 0: # if no samples pass buffer, allow all samples
            buffer_mask = torch.ones_like(buffer_mask, dtype=torch.bool)

        return buffer_mask

    def forward(self, info):

        # Get variables
        x_t = info["noised_inputs"]
        x_t_min = info["targets"]
        output = info["output"]
        assert x_t.shape == x_t_min.shape == output.shape, f"x_0, x_t, x_t_min, and output must have the same shape, but got: x_t.shape: {x_t.shape}, x_t_min.shape: {x_t_min.shape}, output.shape: {output.shape}"
        alpha_t = info["alphas"]
        sigma_t = info["sigmas"]
        alpha_t_min = info["alpha_t_min"]
        sigma_t_min = info["sigma_t_min"]
        assert alpha_t.shape == sigma_t.shape == alpha_t_min.shape == sigma_t_min.shape, f"Number of alphas, sigmas, and min noise levels must match, shape: (batch_size, 1, 1), but got: alpha_t.shape: {alpha_t.shape}, sigma_t.shape: {sigma_t.shape}, alpha_t_min.shape: {alpha_t_min.shape}, sigma_t_min.shape: {sigma_t_min.shape}"

        # Convert model output to x0 prediction
        if self.diffusion_objective == "v":
            x0_pred = alpha_t * x_t - sigma_t * output
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            x0_pred = x_t - sigma_t * output
        else:
            raise ValueError(f"Unknown objective: {self.diffusion_objective}")

        # Compute weighting coefficients, and interpolate model output with current state
        divide_by_zero_epsilon = 1e-8
        coef_model = ((alpha_t_min**2 * sigma_t**2) - (alpha_t**2 * sigma_t_min**2)) / (sigma_t**2 * alpha_t_min + divide_by_zero_epsilon)
        coef_x_t = (alpha_t * sigma_t_min**2) / (alpha_t_min * sigma_t**2 + divide_by_zero_epsilon)
        prediction = (coef_model * x0_pred) + (coef_x_t * x_t)

        # Compare to observation at noise level t_min
        loss_per_sample = F.mse_loss(prediction, x_t_min, reduction='none')

        # Apply ambient weight if enabled
        if self.use_ambient_weight:
            ambient_weight = self.compute_ambient_weight(sigma_t, sigma_t_min) # compute ambient weight
            buffer_mask = self.apply_buffer(sigma_t, sigma_t_min) # apply buffer mask (zeros out samples too close to t_min)
            combined_weight = ambient_weight * buffer_mask.float() # combine: weight * mask (mask zeros out buffered samples)
            weighted_loss = loss_per_sample * combined_weight # apply weight to loss
            loss = weighted_loss.sum() / (buffer_mask.float().sum() + 1e-8) # Reduce: mean over all dimensions, buffered samples contribute 0, so effectively averaged over valid samples
        else:
            loss = loss_per_sample.mean() # standard just mean over all dimensions, reduces to standard MSE loss

        # # -- TEMPORARY DEBUGGING --
        # debug_x0_loss = F.mse_loss(x0_pred, x_t_min)
        # info["debug_x0_loss"] = debug_x0_loss
        # # -- END TEMPORARY DEBUGGING --

        # Decay weight and return loss
        self.decay_weight()
        return self.weight * loss

class LossWithTarget(LossModule):
    def __init__(self, loss_module, input_key: str, target_key: str, name: str, weight: float = 1, decay = 1.0):
        super().__init__(name, weight, decay)

        self.loss_module = loss_module

        self.input_key = input_key
        self.target_key = target_key

    def forward(self, info):
        loss = self.loss_module(info[self.input_key], info[self.target_key])
        self.decay_weight()
        return self.weight * loss

class AuralossLoss(LossWithTarget):
    def __init__(self, loss_module, input_key: str, target_key: str, name: str, weight: float = 1, decay = 1.0):
        super().__init__(loss_module, input_key=input_key, target_key=target_key, name=name, weight=weight, decay=decay)
    def forward(self, info):
        loss = self.loss_module(info[self.target_key], info[self.input_key]) # Enforce wrong order of input and target until we find issue in Auraloss
        self.decay_weight()
        return self.weight * loss

class MultiLoss(nn.Module):
    def __init__(self, losses: tp.List[LossModule]):
        super().__init__()

        self.losses = nn.ModuleList(losses)

    def forward(self, info):
        total_loss = 0

        losses = {}

        for loss_module in self.losses:
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss

        return total_loss, losses

class StereoImageLoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'stereo_image_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        loss = 0.5*(1 - F.cosine_similarity(info[self.key_a], info[self.key_b], dim=1))

        if self.mask_key is not None and self.mask_key in info:
            loss = loss[info[self.mask_key]]

        loss = loss.mean()
        self.decay_weight()
        return self.weight * loss

class TimeDomainMMDLoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0,  name: str = 'time_domain_mmd_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

    def forward(self, info):
        loss = mmd(info[self.key_a], info[self.key_b], bandwidths=[0.0001,0.001,0.01,0.1,1], dim=-1)
        self.decay_weight()
        return self.weight * loss
