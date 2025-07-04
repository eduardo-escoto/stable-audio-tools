# Model Factory Analysis

## Overview

The main model factory in `stable_audio_tools/models/factory.py` provides the core dispatch mechanism for creating models from configuration. This analysis covers the `create_model_from_config()` function and all supported model types.

## Main Factory Function

### `create_model_from_config(model_config)`

**Purpose**: Main entry point for creating models from configuration
**Location**: `stable_audio_tools/models/factory.py:3`

### Configuration Schema

```python
{
    "model_type": str,  # Required - determines which factory to use
    "sample_rate": int,  # Required - audio sample rate
    "sample_size": int,  # Required - audio sample size (for training)
    "audio_channels": int,  # Optional - defaults to 2
    "model": dict,  # Required - model-specific configuration
    "training": dict,  # Optional - training configuration
}
```

### Required Fields

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `model_type` | `str` | Model type identifier | Must be one of supported types |
| `sample_rate` | `int` | Audio sample rate | Must be positive |
| `sample_size` | `int` | Audio sample size | Must be positive |
| `audio_channels` | `int` | Number of audio channels | Optional, defaults to 2 |
| `model` | `dict` | Model-specific configuration | Required, structure depends on model_type |

### Optional Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `training` | `dict` | Training configuration | `None` |

## Supported Model Types

### 1. `autoencoder`
- **Factory**: `stable_audio_tools.models.autoencoders.create_autoencoder_from_config`
- **Purpose**: Standard autoencoder models
- **Config Structure**: See [autoencoder_factory.md](./autoencoder_factory.md)

### 2. `hyperencoder`
- **Factory**: `stable_audio_tools.models.autoencoders.create_hyperencoder_from_config`
- **Purpose**: Hypernetwork-based autoencoder
- **Config Structure**: See [autoencoder_factory.md](./autoencoder_factory.md)

### 3. `diffusion_uncond`
- **Factory**: `stable_audio_tools.models.diffusion.create_diffusion_uncond_from_config`
- **Purpose**: Unconditional diffusion models
- **Config Structure**: See [diffusion_factory.md](./diffusion_factory.md)

### 4. `diffusion_cond`
- **Factory**: `stable_audio_tools.models.diffusion.create_diffusion_cond_from_config`
- **Purpose**: Conditional diffusion models
- **Config Structure**: See [diffusion_factory.md](./diffusion_factory.md)

### 5. `diffusion_cond_inpaint`
- **Factory**: `stable_audio_tools.models.diffusion.create_diffusion_cond_from_config`
- **Purpose**: Conditional diffusion models for inpainting
- **Config Structure**: Same as `diffusion_cond` - See [diffusion_factory.md](./diffusion_factory.md)

### 6. `diffusion_autoencoder`
- **Factory**: `stable_audio_tools.models.autoencoders.create_diffAE_from_config`
- **Purpose**: Diffusion-based autoencoder
- **Config Structure**: See [autoencoder_factory.md](./autoencoder_factory.md)

### 7. `lm`
- **Factory**: `stable_audio_tools.models.lm.create_audio_lm_from_config`
- **Purpose**: Language model for audio
- **Config Structure**: See [language_model_factory.md](./language_model_factory.md)

## Error Handling

- **Missing model_type**: Raises `AssertionError` with message "model_type must be specified in model config"
- **Unknown model_type**: Raises `NotImplementedError` with message "Unknown model type: {model_type}"

## Pydantic Schema Design

### Base Model Config

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from enum import Enum

class ModelType(str, Enum):
    AUTOENCODER = "autoencoder"
    HYPERENCODER = "hyperencoder"
    DIFFUSION_UNCOND = "diffusion_uncond"
    DIFFUSION_COND = "diffusion_cond"
    DIFFUSION_COND_INPAINT = "diffusion_cond_inpaint"
    DIFFUSION_AUTOENCODER = "diffusion_autoencoder"
    LANGUAGE_MODEL = "lm"

class BaseModelConfig(BaseModel):
    """Base configuration for all model types"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    model_type: ModelType = Field(..., description="Type of model to create")
    sample_rate: int = Field(..., gt=0, description="Audio sample rate in Hz")
    sample_size: int = Field(..., gt=0, description="Audio sample size for training")
    audio_channels: int = Field(2, ge=1, le=16, description="Number of audio channels")
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        # Common audio sample rates
        valid_rates = [8000, 16000, 22050, 44100, 48000, 96000, 192000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate {v} not in common rates: {valid_rates}")
        return v
```

### Model-Specific Configuration Union

```python
from typing import Union
from pydantic import Field, discriminator

class ModelConfig(BaseModel):
    """Top-level model configuration with discriminated union"""
    
    # Common fields
    model_type: ModelType
    sample_rate: int
    sample_size: int
    audio_channels: int = 2
    
    # Model-specific configuration
    model: Union[
        AutoencoderModelConfig,
        HyperencoderModelConfig,
        DiffusionUncondModelConfig,
        DiffusionCondModelConfig,
        DiffusionAutoencoderModelConfig,
        LanguageModelConfig,
    ] = Field(..., discriminator='model_type')
    
    # Optional training configuration
    training: Optional[TrainingConfig] = None
```

## Usage Examples

### Autoencoder Example
```json
{
    "model_type": "autoencoder",
    "sample_rate": 44100,
    "sample_size": 1048576,
    "audio_channels": 2,
    "model": {
        "encoder": {...},
        "decoder": {...},
        "bottleneck": {...}
    }
}
```

### Diffusion Model Example
```json
{
    "model_type": "diffusion_cond",
    "sample_rate": 44100,
    "sample_size": 1048576,
    "audio_channels": 2,
    "model": {
        "pretransform": {...},
        "conditioning": {...},
        "diffusion": {...}
    }
}
```

## Dependencies

This factory depends on:
- Individual model factories (autoencoders, diffusion, lm)
- Model implementation modules
- Configuration validation (future Pydantic schemas)

## Migration Notes

- Current implementation uses dictionary access with `.get()` and assertions
- Pydantic migration will provide:
  - Type validation at creation time
  - Better error messages
  - IDE autocomplete support
  - Schema documentation
  - Automatic validation of nested structures 