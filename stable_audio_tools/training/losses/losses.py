import typing as tp
import torch
import math

from torch.nn import functional as F
from torch import nn
from .utils import mmd

import ambient_utils

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

class AmbientDiffusionLoss(LossModule):
    """
    Ambient Diffusion Loss for training with minimum noise level constraints.
    
    This loss implements the ambient diffusion loss from the ambient diffusion paper.
    It requires per-sample minimum noise levels and computes the loss based on
    predictions at the minimum noise level rather than at the sampled timestep.
    
    Args:
        output_key: Key in info dict for model output (D_xn prediction)
        x_key: Key in info dict for clean input data (x)
        noise_key: Key in info dict for noise tensor
        sigma_t_key: Key in info dict for sigma_t values (noise level at timestep t)
        min_noise_levels_key: Key in info dict for minimum noise levels per sample (sigma_tn)
        weight: Loss weight
        mask_key: Optional key for padding mask
        name: Loss name
        decay: Weight decay factor
        sigma_data: Expected standard deviation of training data (default 0.5 for normalized data)
    """
    def __init__(
        self,
        output_key: str,
        x_key: str,
        noise_key: str,
        sigma_t_key: str,
        min_noise_levels_key: str,
        weight: float = 1.0,
        mask_key: str = None,
        name: str = 'ambient_diffusion_loss',
        decay: float = 1.0,
        sigma_data: float = 0.5
    ):
        super().__init__(name=name, weight=weight, decay=decay)
        
        self.output_key = output_key
        self.x_key = x_key
        self.noise_key = noise_key
        self.sigma_t_key = sigma_t_key
        self.min_noise_levels_key = min_noise_levels_key
        self.mask_key = mask_key
        self.sigma_data = sigma_data
    
    def _get_alphas_sigmas(self, t):
        """Convert timesteps to alpha and sigma values (cosine schedule)."""
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
    
    def forward(self, info):
        """
        Compute ambient diffusion loss.
        
        The loss computation follows the ambient diffusion paper:
        1. Convert min_noise_levels from timestep space to sigma space
        2. Compute ambient weights based on sigma_t and sigma_tn
        3. Create x_tn = x * alpha_tn + noise * sigma_tn (noise at minimum level)
        4. Convert model output D_xn to D_xn_tn (prediction at minimum noise level)
        5. Compute loss as ambient_weight * (D_xn_tn - x_tn)^2
        """
        # Extract inputs from info dict
        D_xn = info[self.output_key]  # Model output (v-prediction format)
        x = info[self.x_key]  # Clean input data
        noise = info[self.noise_key]  # Noise tensor used to create x_t
        sigma_t = info[self.sigma_t_key]  # sigma_t values (already in sigma space, shape: (batch_size,))
        min_noise_levels = info[self.min_noise_levels_key]  # min_noise_levels in timestep space [0,1], shape: (batch_size,)
        x_t = info.get("noised_inputs")  # x_t for ambient_utils conversion
        
        # 1. Convert min_noise_levels from timestep space to sigma space
        # This matches the starter code's assumption that both sigma_t and sigma_tn are in sigma space
        _, sigma_tn = self._get_alphas_sigmas(min_noise_levels)  # Convert t → sigma, shape: (batch_size,)
        alpha_tn, _ = self._get_alphas_sigmas(min_noise_levels)  # Also need alpha_tn for cosine schedule
        
        # Ensure sigma_t and sigma_tn are on the same device and have compatible shapes
        device = D_xn.device
        sigma_t = sigma_t.to(device)
        sigma_tn = sigma_tn.to(device)
        alpha_tn = alpha_tn.to(device)
        
        # 2. Reconstruct x_tn using the same noise tensor (faithful to starter code)
        # In cosine schedule: x_tn = x * alpha_tn + noise * sigma_tn
        # This is equivalent to starter code's x_tn = x + sigma_tn * noise when alpha_tn ≈ 1
        sigma_tn_spatial = sigma_tn[:, None, None]  # Add spatial dimensions for broadcasting
        alpha_tn_spatial = alpha_tn[:, None, None]
        x_tn = x * alpha_tn_spatial + noise * sigma_tn_spatial
        
        # 3. Compute ambient weights (faithful to starter code formulas)
        # ambient_factor = (sigma_t^4) / ((sigma_t^2 - sigma_tn^2)^2)
        sigma_t_sq = sigma_t ** 2
        sigma_tn_sq = sigma_tn ** 2
        ambient_factor = (sigma_t ** 4) / ((sigma_t_sq - sigma_tn_sq) ** 2 + 1e-8)  # Add small epsilon for numerical stability
        
        # edm_weight = (sigma_data^2 + sigma_t^2) / ((sigma_data^2) * (sigma_t^2))
        sigma_data_sq = self.sigma_data ** 2
        edm_weight = (sigma_data_sq + sigma_t_sq) / (sigma_data_sq * sigma_t_sq + 1e-8)
        
        # ambient_weight = ambient_factor * edm_weight
        ambient_weight = ambient_factor * edm_weight  # Shape: (batch_size,)
        ambient_weight_spatial = ambient_weight[:, None, None]  # Add spatial dimensions
        
        # 4. Convert model output D_xn to D_xn_tn (prediction at minimum noise level)
        # ambient_utils expects x_t, sigma_t, sigma_tn
        # x_t should be the noised input at timestep t
        if x_t is None:
            # Reconstruct x_t if not provided (shouldn't happen, but handle gracefully)
            # We'd need alphas and sigmas for t, but we don't have t directly
            # For now, raise an error
            raise ValueError(
                "noised_inputs (x_t) is required in info dict for ambient loss. "
                "Make sure it's added to loss_info when ambient=True."
            )
        
        # Convert sigma_t and sigma_tn to match expected shapes for ambient_utils
        # ambient_utils likely expects them as tensors with spatial dimensions or as scalars per sample
        # Check the function signature, but typically it handles broadcasting
        D_xn_tn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(
            D_xn, x_t, sigma_t, sigma_tn
        )
        
        # 5. Compute loss: ambient_weight * (D_xn_tn - x_tn)^2
        loss = ambient_weight_spatial * ((D_xn_tn - x_tn) ** 2)
        
        # Apply masking if provided
        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]
            if mask.ndim == 2 and loss.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[1] != loss.shape[1]:
                mask = mask.repeat(1, loss.shape[1], 1)
            loss = loss[mask]
        
        # Mean reduction
        loss = loss.mean()
        
        self.decay_weight()
        return self.weight * loss
