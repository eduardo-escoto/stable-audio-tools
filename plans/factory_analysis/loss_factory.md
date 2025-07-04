# Loss Factory Analysis

## Overview

The loss factory in `stable_audio_tools/training/autoencoders.py` creates loss modules based on bottleneck types. This factory automatically configures appropriate loss functions for different autoencoder bottleneck architectures.

## Factory Function

### `create_loss_modules_from_bottleneck(bottleneck, loss_config)`

**Purpose**: Creates loss modules based on bottleneck type
**Location**: `stable_audio_tools/training/autoencoders.py:877`

## Function Parameters

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `bottleneck` | `nn.Module` | Bottleneck module instance | Must be supported bottleneck type |
| `loss_config` | `dict` | Loss configuration | Must contain bottleneck weights configuration |

## Configuration Schema

### Loss Configuration Structure
```python
{
    "bottleneck": {
        "weights": {
            "kl": float,  # Optional - KL divergence weight
            "mmd": float,  # Optional - MMD loss weight
            # Other bottleneck-specific weights
        }
    }
}
```

## Supported Bottleneck Types

### 1. VAE Bottlenecks
- **Types**: `VAEBottleneck`, `DACRVQVAEBottleneck`, `RVQVAEBottleneck`
- **Loss**: KL Divergence Loss
- **Purpose**: Regularize latent distribution in variational autoencoders

#### Generated Loss
```python
ValueLoss(
    key="kl",
    weight=kl_weight,  # From loss_config["bottleneck"]["weights"]["kl"]
    name="kl_loss"
)
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `kl_weight` | `float` | `1e-6` | KL divergence weight | Must be non-negative |

### 2. RVQ Bottlenecks
- **Types**: `RVQBottleneck`, `RVQVAEBottleneck`
- **Loss**: Quantizer Loss
- **Purpose**: Regularize residual vector quantization

#### Generated Loss
```python
ValueLoss(
    key="quantizer_loss",
    weight=1.0,  # Fixed weight
    name="quantizer_loss"
)
```

### 3. DAC RVQ Bottlenecks
- **Types**: `DACRVQBottleneck`, `DACRVQVAEBottleneck`
- **Losses**: Codebook Loss and Commitment Loss
- **Purpose**: Regularize DAC-style residual vector quantization

#### Generated Losses
```python
# Codebook Loss
ValueLoss(
    key="vq/codebook_loss",
    weight=1.0,  # Fixed weight
    name="codebook_loss"
)

# Commitment Loss
ValueLoss(
    key="vq/commitment_loss",
    weight=0.25,  # Fixed weight
    name="commitment_loss"
)
```

### 4. Wasserstein Bottlenecks
- **Type**: `WassersteinBottleneck`
- **Loss**: Maximum Mean Discrepancy (MMD) Loss
- **Purpose**: Regularize Wasserstein autoencoder latent distribution

#### Generated Loss
```python
ValueLoss(
    key="mmd",
    weight=mmd_weight,  # From loss_config["bottleneck"]["weights"]["mmd"]
    name="mmd_loss"
)
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `mmd_weight` | `float` | `100` | MMD loss weight | Must be non-negative |

## Loss Types Summary

| Bottleneck Type | Generated Losses | Default Weights | Configuration Keys |
|-----------------|------------------|-----------------|-------------------|
| `VAEBottleneck` | KL Divergence | `1e-6` | `bottleneck.weights.kl` |
| `DACRVQVAEBottleneck` | KL + Codebook + Commitment | `1e-6`, `1.0`, `0.25` | `bottleneck.weights.kl` |
| `RVQVAEBottleneck` | KL + Quantizer | `1e-6`, `1.0` | `bottleneck.weights.kl` |
| `RVQBottleneck` | Quantizer | `1.0` | None |
| `DACRVQBottleneck` | Codebook + Commitment | `1.0`, `0.25` | None |
| `WassersteinBottleneck` | MMD | `100` | `bottleneck.weights.mmd` |

## Error Handling

### Configuration Access Errors
- **Missing weights config**: Uses default values if `loss_config["bottleneck"]["weights"]` path doesn't exist
- **Missing specific weight**: Uses default value for that specific weight
- **Exception handling**: Catches all exceptions with bare `except:` clause and uses defaults

## Return Type

### Loss Modules List
- **Type**: `List[ValueLoss]`
- **Contents**: List of configured loss modules
- **Purpose**: Loss modules to be added to training wrapper

## ValueLoss Class

The factory creates `ValueLoss` instances with the following structure:

```python
class ValueLoss:
    def __init__(self, key: str, weight: float, name: str):
        self.key = key        # Key to extract loss value from model output
        self.weight = weight  # Loss weight for combining with other losses
        self.name = name      # Human-readable loss name for logging
```

## Pydantic Schema Design

### Base Configuration

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum

class BottleneckType(str, Enum):
    VAE = "VAEBottleneck"
    DAC_RVQ_VAE = "DACRVQVAEBottleneck"
    RVQ_VAE = "RVQVAEBottleneck"
    RVQ = "RVQBottleneck"
    DAC_RVQ = "DACRVQBottleneck"
    WASSERSTEIN = "WassersteinBottleneck"

class BottleneckWeights(BaseModel):
    """Configuration for bottleneck loss weights"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    kl: Optional[float] = Field(1e-6, description="KL divergence weight", ge=0)
    mmd: Optional[float] = Field(100.0, description="MMD loss weight", ge=0)

class BottleneckLossConfig(BaseModel):
    """Configuration for bottleneck losses"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    weights: BottleneckWeights = Field(default_factory=BottleneckWeights, description="Loss weights")

class AutoencoderLossConfig(BaseModel):
    """Configuration for autoencoder losses"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    bottleneck: BottleneckLossConfig = Field(default_factory=BottleneckLossConfig, description="Bottleneck loss configuration")
    # Additional loss configurations can be added here

class ValueLossConfig(BaseModel):
    """Configuration for a single value loss"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    key: str = Field(..., description="Key to extract loss value from model output")
    weight: float = Field(..., description="Loss weight", ge=0)
    name: str = Field(..., description="Human-readable loss name")
```

### Loss Factory Result Schema

```python
from typing import List, Union

class LossFactoryResult(BaseModel):
    """Result from loss factory"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    losses: List[ValueLossConfig] = Field(..., description="Generated loss configurations")
    bottleneck_type: BottleneckType = Field(..., description="Detected bottleneck type")
    
    def get_total_weight(self) -> float:
        """Calculate total weight of all losses"""
        return sum(loss.weight for loss in self.losses)
    
    def get_loss_by_name(self, name: str) -> Optional[ValueLossConfig]:
        """Get loss configuration by name"""
        for loss in self.losses:
            if loss.name == name:
                return loss
        return None
```

## Usage Examples

### VAE Autoencoder Loss Configuration
```json
{
    "bottleneck": {
        "weights": {
            "kl": 1e-5
        }
    }
}
```

**Generated Losses**: `[ValueLoss(key="kl", weight=1e-5, name="kl_loss")]`

### DAC RVQ VAE Loss Configuration
```json
{
    "bottleneck": {
        "weights": {
            "kl": 1e-4
        }
    }
}
```

**Generated Losses**:
```python
[
    ValueLoss(key="kl", weight=1e-4, name="kl_loss"),
    ValueLoss(key="quantizer_loss", weight=1.0, name="quantizer_loss"),
    ValueLoss(key="vq/codebook_loss", weight=1.0, name="codebook_loss"),
    ValueLoss(key="vq/commitment_loss", weight=0.25, name="commitment_loss")
]
```

### Wasserstein Autoencoder Loss Configuration
```json
{
    "bottleneck": {
        "weights": {
            "mmd": 50.0
        }
    }
}
```

**Generated Losses**: `[ValueLoss(key="mmd", weight=50.0, name="mmd_loss")]`

### Complete Autoencoder Training Configuration
```json
{
    "model_type": "autoencoder",
    "training": {
        "learning_rate": 1e-4,
        "loss_configs": {
            "bottleneck": {
                "weights": {
                    "kl": 1e-5,
                    "mmd": 75.0
                }
            },
            "reconstruction": {
                "weight": 1.0,
                "type": "l2"
            },
            "adversarial": {
                "weight": 0.1,
                "type": "hinge"
            }
        }
    }
}
```

## Factory Processing Logic

### Bottleneck Type Detection
The factory uses `isinstance()` checks to determine bottleneck type:

```python
def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []
    
    # VAE-based bottlenecks
    if isinstance(bottleneck, (VAEBottleneck, DACRVQVAEBottleneck, RVQVAEBottleneck)):
        kl_weight = get_weight(loss_config, "kl", default=1e-6)
        losses.append(ValueLoss(key="kl", weight=kl_weight, name="kl_loss"))
    
    # RVQ-based bottlenecks
    if isinstance(bottleneck, (RVQBottleneck, RVQVAEBottleneck)):
        losses.append(ValueLoss(key="quantizer_loss", weight=1.0, name="quantizer_loss"))
    
    # DAC RVQ bottlenecks
    if isinstance(bottleneck, (DACRVQBottleneck, DACRVQVAEBottleneck)):
        losses.append(ValueLoss(key="vq/codebook_loss", weight=1.0, name="codebook_loss"))
        losses.append(ValueLoss(key="vq/commitment_loss", weight=0.25, name="commitment_loss"))
    
    # Wasserstein bottlenecks
    if isinstance(bottleneck, WassersteinBottleneck):
        mmd_weight = get_weight(loss_config, "mmd", default=100)
        losses.append(ValueLoss(key="mmd", weight=mmd_weight, name="mmd_loss"))
    
    return losses
```

## Dependencies

The loss factory depends on:
- `stable_audio_tools.models.bottleneck` - Bottleneck implementations
- `stable_audio_tools.training.losses.losses` - ValueLoss class
- Model output structure that provides the required loss keys

## Migration Notes

### Current Implementation Issues
- Bare `except:` clause catches all exceptions, making debugging difficult
- Hard-coded loss weights for some bottleneck types
- No validation of loss configuration structure
- Uses `isinstance()` checks that could be fragile with inheritance

### Pydantic Migration Benefits
- Proper validation of loss configuration structure
- Type-safe loss weight specifications
- Better error messages for configuration issues
- Documentation of all loss types and their purposes
- Elimination of silent failures due to configuration access errors
- Support for additional loss types through extensible schema design
- Validation of loss weight ranges and relationships 