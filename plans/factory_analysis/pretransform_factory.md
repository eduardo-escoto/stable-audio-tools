# Pretransform Factory Analysis

## Overview

The pretransform factory in `stable_audio_tools/models/factory.py` creates preprocessing transforms applied before the main model processing. This analysis covers `create_pretransform_from_config()` and all supported pretransform types.

## Factory Function

### `create_pretransform_from_config(pretransform_config, sample_rate)`

**Purpose**: Creates pretransform modules from configuration
**Location**: `stable_audio_tools/models/factory.py:30`

### Function Signature
```python
def create_pretransform_from_config(pretransform_config, sample_rate):
```

### Configuration Schema

```python
{
    "type": str,  # Required - pretransform type identifier
    "config": dict,  # Required - pretransform-specific parameters
    "scale": float,  # Optional - scaling factor (autoencoder only)
    "model_half": bool,  # Optional - use half precision (autoencoder only)
    "iterate_batch": bool,  # Optional - iterate batch (autoencoder only)
    "chunked": bool,  # Optional - use chunked processing (autoencoder only)
    "enable_grad": bool,  # Optional - enable gradients
}
```

### Common Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `type` | `str` | Pretransform type identifier | **Required** | Must be one of supported types |
| `config` | `dict` | Type-specific configuration | **Required** | Structure depends on type |
| `enable_grad` | `bool` | Whether to enable gradients | `False` | Boolean |

### Autoencoder-Specific Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `scale` | `float` | Scaling factor | `1.0` | Must be positive |
| `model_half` | `bool` | Use half precision | `False` | Boolean |
| `iterate_batch` | `bool` | Iterate batch processing | `False` | Boolean |
| `chunked` | `bool` | Use chunked processing | `False` | Boolean |

## Supported Pretransform Types

### 1. `autoencoder` - Autoencoder Pretransform
- **Class**: `AutoencoderPretransform`
- **Purpose**: Uses another autoencoder as a pretransform
- **Special**: Creates nested autoencoder configuration

#### Configuration
```python
{
    "type": "autoencoder",
    "config": {
        # Complete autoencoder model configuration
        "encoder": {...},  # Encoder configuration
        "decoder": {...},  # Decoder configuration
        "bottleneck": {...},  # Optional bottleneck configuration
        "latent_dim": int,  # Required
        "downsampling_ratio": int,  # Required
        "io_channels": int,  # Required
        # Additional autoencoder parameters
    },
    "scale": float,  # Optional, default: 1.0
    "model_half": bool,  # Optional, default: False
    "iterate_batch": bool,  # Optional, default: False
    "chunked": bool,  # Optional, default: False
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `config` | `dict` | **Required** | Complete autoencoder model config | Must be valid autoencoder config |
| `scale` | `float` | `1.0` | Scaling factor | Must be positive |
| `model_half` | `bool` | `False` | Use half precision | Boolean |
| `iterate_batch` | `bool` | `False` | Iterate batch processing | Boolean |
| `chunked` | `bool` | `False` | Use chunked processing | Boolean |

**Note**: The factory creates a fake top-level config with `sample_rate` to pass to the autoencoder constructor.

### 2. `wavelet` - Wavelet Pretransform
- **Class**: `WaveletPretransform`
- **Purpose**: Wavelet decomposition pretransform
- **Module**: `stable_audio_tools.models.pretransforms`

#### Configuration
```python
{
    "type": "wavelet",
    "config": {
        "channels": int,  # Required - number of channels
        "levels": int,  # Required - decomposition levels
        "wavelet": str,  # Required - wavelet type
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `channels` | `int` | **Required** | Number of channels | Must be positive |
| `levels` | `int` | **Required** | Decomposition levels | Must be positive |
| `wavelet` | `str` | **Required** | Wavelet type | Must be valid wavelet name |

### 3. `pqmf` - Pseudo-QMF Pretransform
- **Class**: `PQMFPretransform`
- **Purpose**: Pseudo-quadrature mirror filter bank
- **Module**: `stable_audio_tools.models.pretransforms`

#### Configuration
```python
{
    "type": "pqmf",
    "config": {
        # PQMF-specific parameters
        # Passed directly to PQMFPretransform constructor
    }
}
```

**Note**: All parameters in `config` are passed directly to the `PQMFPretransform` constructor.

### 4. `dac_pretrained` - Pretrained DAC Pretransform
- **Class**: `PretrainedDACPretransform`
- **Purpose**: Uses a pretrained DAC model as pretransform
- **Module**: `stable_audio_tools.models.pretransforms`

#### Configuration
```python
{
    "type": "dac_pretrained",
    "config": {
        # DAC-specific parameters
        # Passed directly to PretrainedDACPretransform constructor
    }
}
```

**Note**: All parameters in `config` are passed directly to the `PretrainedDACPretransform` constructor.

### 5. `audiocraft_pretrained` - Audiocraft Compression Pretransform
- **Class**: `AudiocraftCompressionPretransform`
- **Purpose**: Uses Audiocraft's compression model as pretransform
- **Module**: `stable_audio_tools.models.pretransforms`

#### Configuration
```python
{
    "type": "audiocraft_pretrained",
    "config": {
        # Audiocraft-specific parameters
        # Passed directly to AudiocraftCompressionPretransform constructor
    }
}
```

**Note**: All parameters in `config` are passed directly to the `AudiocraftCompressionPretransform` constructor.

### 6. `patched` - Patched Pretransform
- **Class**: `PatchedPretransform`
- **Purpose**: Applies patching to the input
- **Module**: `stable_audio_tools.models.pretransforms`

#### Configuration
```python
{
    "type": "patched",
    "config": {
        # Patching-specific parameters
        # Passed directly to PatchedPretransform constructor
    }
}
```

**Note**: All parameters in `config` are passed directly to the `PatchedPretransform` constructor.

## Post-Processing

After creating the pretransform, the factory applies these common settings:

1. **Gradient Control**: Sets `enable_grad` attribute based on configuration
2. **Evaluation Mode**: Calls `pretransform.eval()`
3. **Gradient Requirement**: Calls `pretransform.requires_grad_(pretransform.enable_grad)`

## Error Handling

- **Missing type**: Raises `AssertionError` with message "type must be specified in pretransform config"
- **Unknown type**: Raises `NotImplementedError` with message "Unknown pretransform type: {type}"

## Pydantic Schema Design

### Base Pretransform Config

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Union
from enum import Enum

class PretransformType(str, Enum):
    AUTOENCODER = "autoencoder"
    WAVELET = "wavelet"
    PQMF = "pqmf"
    DAC_PRETRAINED = "dac_pretrained"
    AUDIOCRAFT_PRETRAINED = "audiocraft_pretrained"
    PATCHED = "patched"

class BasePretransformConfig(BaseModel):
    """Base configuration for pretransform modules"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: PretransformType = Field(..., description="Type of pretransform")
    config: Dict[str, Any] = Field(..., description="Type-specific configuration")
    enable_grad: bool = Field(False, description="Whether to enable gradients")
```

### Specific Pretransform Configs

```python
class AutoencoderPretransformConfig(BasePretransformConfig):
    type: Literal[PretransformType.AUTOENCODER] = PretransformType.AUTOENCODER
    config: Dict[str, Any] = Field(..., description="Complete autoencoder model configuration")
    scale: float = Field(1.0, gt=0, description="Scaling factor")
    model_half: bool = Field(False, description="Use half precision")
    iterate_batch: bool = Field(False, description="Iterate batch processing")
    chunked: bool = Field(False, description="Use chunked processing")
    
    @validator('config')
    def validate_autoencoder_config(cls, v):
        # Validate that config contains required autoencoder fields
        required_fields = ['encoder', 'decoder', 'latent_dim', 'downsampling_ratio', 'io_channels']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field '{field}' in autoencoder config")
        return v

class WaveletPretransformConfig(BasePretransformConfig):
    type: Literal[PretransformType.WAVELET] = PretransformType.WAVELET
    config: Dict[str, Any] = Field(..., description="Wavelet configuration")
    
    @validator('config')
    def validate_wavelet_config(cls, v):
        required_fields = ['channels', 'levels', 'wavelet']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field '{field}' in wavelet config")
        
        if 'channels' in v and v['channels'] <= 0:
            raise ValueError("channels must be positive")
        if 'levels' in v and v['levels'] <= 0:
            raise ValueError("levels must be positive")
        
        return v

class PQMFPretransformConfig(BasePretransformConfig):
    type: Literal[PretransformType.PQMF] = PretransformType.PQMF
    config: Dict[str, Any] = Field(..., description="PQMF configuration")

class DACPretrainedPretransformConfig(BasePretransformConfig):
    type: Literal[PretransformType.DAC_PRETRAINED] = PretransformType.DAC_PRETRAINED
    config: Dict[str, Any] = Field(..., description="DAC configuration")

class AudiocraftPretrainedPretransformConfig(BasePretransformConfig):
    type: Literal[PretransformType.AUDIOCRAFT_PRETRAINED] = PretransformType.AUDIOCRAFT_PRETRAINED
    config: Dict[str, Any] = Field(..., description="Audiocraft configuration")

class PatchedPretransformConfig(BasePretransformConfig):
    type: Literal[PretransformType.PATCHED] = PretransformType.PATCHED
    config: Dict[str, Any] = Field(..., description="Patching configuration")
```

### Discriminated Union

```python
from typing import Union
from pydantic import Field

PretransformConfig = Union[
    AutoencoderPretransformConfig,
    WaveletPretransformConfig,
    PQMFPretransformConfig,
    DACPretrainedPretransformConfig,
    AudiocraftPretrainedPretransformConfig,
    PatchedPretransformConfig,
] = Field(..., discriminator='type')
```

## Usage Examples

### Autoencoder Pretransform
```json
{
    "type": "autoencoder",
    "config": {
        "encoder": {
            "type": "seanet",
            "config": {...}
        },
        "decoder": {
            "type": "seanet",
            "config": {...}
        },
        "latent_dim": 64,
        "downsampling_ratio": 320,
        "io_channels": 2
    },
    "scale": 1.0,
    "model_half": false,
    "chunked": true
}
```

### Wavelet Pretransform
```json
{
    "type": "wavelet",
    "config": {
        "channels": 2,
        "levels": 6,
        "wavelet": "db4"
    }
}
```

### PQMF Pretransform
```json
{
    "type": "pqmf",
    "config": {
        "attn": 100,
        "n_bands": 16
    }
}
```

## Dependencies

The pretransform factory depends on:
- `stable_audio_tools.models.pretransforms` - Pretransform implementations
- `stable_audio_tools.models.factory.create_autoencoder_from_config` - For autoencoder pretransforms
- Various external libraries (wavelet, DAC, Audiocraft)

## Migration Notes

- Current implementation uses direct parameter passing to constructors
- The autoencoder pretransform uses a "hack" to create fake config with sample_rate
- Pydantic migration will provide:
  - Proper validation of all parameters
  - Type checking for all configuration fields
  - Better error messages for invalid configurations
  - Documentation of all parameters
  - Cleaner handling of nested configurations (autoencoder) 