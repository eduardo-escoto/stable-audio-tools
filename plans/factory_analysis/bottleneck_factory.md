# Bottleneck Factory Analysis

## Overview

The bottleneck factory in `stable_audio_tools/models/factory.py` creates different types of bottleneck modules used in autoencoder architectures. This analysis covers `create_bottleneck_from_config()` and all supported bottleneck types.

## Factory Function

### `create_bottleneck_from_config(bottleneck_config)`

**Purpose**: Creates bottleneck modules from configuration
**Location**: `stable_audio_tools/models/factory.py:90`

### Configuration Schema

```python
{
    "type": str,  # Required - bottleneck type identifier
    "config": dict,  # Optional - bottleneck-specific parameters
    "requires_grad": bool,  # Optional - whether parameters require gradients
}
```

### Common Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `type` | `str` | Bottleneck type identifier | **Required** | Must be one of supported types |
| `config` | `dict` | Type-specific configuration | `{}` | Structure depends on type |
| `requires_grad` | `bool` | Whether parameters require gradients | `True` | Boolean |

## Supported Bottleneck Types

### 1. `tanh` - Tanh Bottleneck
- **Class**: `TanhBottleneck`
- **Purpose**: Applies tanh activation with scaling
- **Discrete**: `False`

#### Configuration
```python
{
    "type": "tanh",
    "config": {
        "scale": float  # Optional, default: 1.0
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `scale` | `float` | `1.0` | Scaling factor for tanh | Must be positive |

### 2. `vae` - VAE Bottleneck
- **Class**: `VAEBottleneck`
- **Purpose**: Variational Autoencoder bottleneck with sampling
- **Discrete**: `False`

#### Configuration
```python
{
    "type": "vae",
    "config": {}  # No parameters - uses defaults
}
```

**Note**: VAE bottleneck has no configurable parameters.

### 3. `rvq` - Residual Vector Quantization
- **Class**: `RVQBottleneck`
- **Purpose**: Residual vector quantization bottleneck
- **Discrete**: `True`

#### Configuration
```python
{
    "type": "rvq",
    "config": {
        "dim": int,  # Optional, default: 128
        "codebook_size": int,  # Optional, default: 1024
        "num_quantizers": int,  # Optional, default: 8
        "decay": float,  # Optional, default: 0.99
        "kmeans_init": bool,  # Optional, default: True
        "kmeans_iters": int,  # Optional, default: 50
        "threshold_ema_dead_code": int,  # Optional, default: 2
        # Additional parameters passed to ResidualVQ
    }
}
```

#### Default Parameters
| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `dim` | `int` | `128` | Dimension of vectors | Must be positive |
| `codebook_size` | `int` | `1024` | Size of each codebook | Must be positive |
| `num_quantizers` | `int` | `8` | Number of quantizers | Must be positive |
| `decay` | `float` | `0.99` | EMA decay rate | Must be between 0 and 1 |
| `kmeans_init` | `bool` | `True` | Use k-means initialization | Boolean |
| `kmeans_iters` | `int` | `50` | Number of k-means iterations | Must be positive |
| `threshold_ema_dead_code` | `int` | `2` | Threshold for dead code | Must be positive |

### 4. `dac_rvq` - DAC Residual Vector Quantization
- **Class**: `DACRVQBottleneck`
- **Purpose**: DAC-style residual vector quantization
- **Discrete**: `True`

#### Configuration
```python
{
    "type": "dac_rvq",
    "config": {
        "quantize_on_decode": bool,  # Optional, default: False
        "noise_augment_dim": int,  # Optional, default: 0
        # DAC-specific quantizer parameters
        "n_codebooks": int,  # Required
        "codebook_size": int,  # Required
        # Additional DAC parameters
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `quantize_on_decode` | `bool` | `False` | Whether to quantize on decode | Boolean |
| `noise_augment_dim` | `int` | `0` | Dimension of noise augmentation | Must be non-negative |
| `n_codebooks` | `int` | **Required** | Number of codebooks | Must be positive |
| `codebook_size` | `int` | **Required** | Size of each codebook | Must be positive |

### 5. `rvq_vae` - RVQ + VAE Bottleneck
- **Class**: `RVQVAEBottleneck`
- **Purpose**: Combines VAE sampling with RVQ quantization
- **Discrete**: `True`

#### Configuration
```python
{
    "type": "rvq_vae",
    "config": {
        "dim": int,  # Optional, default: 128
        "codebook_size": int,  # Optional, default: 1024
        "num_quantizers": int,  # Optional, default: 8
        "decay": float,  # Optional, default: 0.99
        "kmeans_init": bool,  # Optional, default: True
        "kmeans_iters": int,  # Optional, default: 50
        "threshold_ema_dead_code": int,  # Optional, default: 2
    }
}
```

**Note**: Uses same default parameters as `rvq` type.

### 6. `dac_rvq_vae` - DAC RVQ + VAE Bottleneck
- **Class**: `DACRVQVAEBottleneck`
- **Purpose**: DAC-style RVQ with VAE sampling
- **Discrete**: `True`

#### Configuration
```python
{
    "type": "dac_rvq_vae",
    "config": {
        "quantize_on_decode": bool,  # Optional, default: False
        # DAC-specific quantizer parameters
        "n_codebooks": int,  # Required
        "codebook_size": int,  # Required
        # Additional DAC parameters
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `quantize_on_decode` | `bool` | `False` | Whether to quantize on decode | Boolean |
| `n_codebooks` | `int` | **Required** | Number of codebooks | Must be positive |
| `codebook_size` | `int` | **Required** | Size of each codebook | Must be positive |

### 7. `l2_norm` - L2 Normalization Bottleneck
- **Class**: `L2Bottleneck`
- **Purpose**: L2 normalization bottleneck
- **Discrete**: `False`

#### Configuration
```python
{
    "type": "l2_norm",
    "config": {}  # No parameters
}
```

**Note**: L2 bottleneck has no configurable parameters.

### 8. `wasserstein` - Wasserstein Bottleneck
- **Class**: `WassersteinBottleneck`
- **Purpose**: Wasserstein distance-based bottleneck
- **Discrete**: `False`

#### Configuration
```python
{
    "type": "wasserstein",
    "config": {
        "noise_augment_dim": int,  # Optional, default: 0
        "bypass_mmd": bool,  # Optional, default: False
        "use_tanh": bool,  # Optional, default: False
        "tanh_scale": float,  # Optional, default: 5.0
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `noise_augment_dim` | `int` | `0` | Dimension of noise augmentation | Must be non-negative |
| `bypass_mmd` | `bool` | `False` | Whether to bypass MMD computation | Boolean |
| `use_tanh` | `bool` | `False` | Whether to use tanh activation | Boolean |
| `tanh_scale` | `float` | `5.0` | Scaling factor for tanh | Must be positive |

### 9. `fsq` - Finite Scalar Quantization
- **Class**: `FSQBottleneck`
- **Purpose**: Finite scalar quantization bottleneck
- **Discrete**: `True`

#### Configuration
```python
{
    "type": "fsq",
    "config": {
        "noise_augment_dim": int,  # Optional, default: 0
        "levels": list,  # Required - quantization levels
        "num_codebooks": int,  # Optional, default: 1
        # Additional FSQ parameters
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `noise_augment_dim` | `int` | `0` | Dimension of noise augmentation | Must be non-negative |
| `levels` | `list` | **Required** | Quantization levels | Must be list of positive integers |
| `num_codebooks` | `int` | `1` | Number of codebooks | Must be positive |

### 10. `dithered_fsq` - Dithered Finite Scalar Quantization
- **Class**: `DitheredFSQBottleneck`
- **Purpose**: Dithered finite scalar quantization
- **Discrete**: `True`

#### Configuration
```python
{
    "type": "dithered_fsq",
    "config": {
        "dim": int,  # Required - dimension
        "levels": Union[int, List[int]],  # Required - quantization levels
        "num_codebooks": int,  # Optional, default: 1
        "dither_inference": bool,  # Optional, default: True
        "noise_dropout": float,  # Optional, default: 0.05
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `dim` | `int` | **Required** | Dimension of vectors | Must be positive |
| `levels` | `Union[int, List[int]]` | **Required** | Quantization levels | Int or list of positive integers |
| `num_codebooks` | `int` | `1` | Number of codebooks | Must be positive |
| `dither_inference` | `bool` | `True` | Whether to use dithering at inference | Boolean |
| `noise_dropout` | `float` | `0.05` | Noise dropout probability | Must be between 0 and 1 |

## Error Handling

- **Missing type**: Raises `AssertionError` with message "type must be specified in bottleneck config"
- **Unknown type**: Raises `NotImplementedError` with message "Unknown bottleneck type: {type}"

## Pydantic Schema Design

### Base Bottleneck Config

```python
from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, Any, Union, List
from enum import Enum

class BottleneckType(str, Enum):
    TANH = "tanh"
    VAE = "vae"
    RVQ = "rvq"
    DAC_RVQ = "dac_rvq"
    RVQ_VAE = "rvq_vae"
    DAC_RVQ_VAE = "dac_rvq_vae"
    L2_NORM = "l2_norm"
    WASSERSTEIN = "wasserstein"
    FSQ = "fsq"
    DITHERED_FSQ = "dithered_fsq"

class BaseBottleneckConfig(BaseModel):
    """Base configuration for bottleneck modules"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: BottleneckType = Field(..., description="Type of bottleneck")
    requires_grad: bool = Field(True, description="Whether parameters require gradients")
```

### Specific Bottleneck Configs

```python
class TanhBottleneckConfig(BaseBottleneckConfig):
    type: Literal[BottleneckType.TANH] = BottleneckType.TANH
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('config')
    def validate_tanh_config(cls, v):
        if 'scale' in v:
            assert v['scale'] > 0, "scale must be positive"
        return v

class RVQBottleneckConfig(BaseBottleneckConfig):
    type: Literal[BottleneckType.RVQ] = BottleneckType.RVQ
    config: Dict[str, Any] = Field(default_factory=lambda: {
        "dim": 128,
        "codebook_size": 1024,
        "num_quantizers": 8,
        "decay": 0.99,
        "kmeans_init": True,
        "kmeans_iters": 50,
        "threshold_ema_dead_code": 2,
    })
    
    @validator('config')
    def validate_rvq_config(cls, v):
        if 'dim' in v:
            assert v['dim'] > 0, "dim must be positive"
        if 'codebook_size' in v:
            assert v['codebook_size'] > 0, "codebook_size must be positive"
        if 'num_quantizers' in v:
            assert v['num_quantizers'] > 0, "num_quantizers must be positive"
        if 'decay' in v:
            assert 0 <= v['decay'] <= 1, "decay must be between 0 and 1"
        return v

# ... Additional specific configs for each type
```

### Discriminated Union

```python
from typing import Union
from pydantic import Field

BottleneckConfig = Union[
    TanhBottleneckConfig,
    VAEBottleneckConfig,
    RVQBottleneckConfig,
    DACRVQBottleneckConfig,
    RVQVAEBottleneckConfig,
    DACRVQVAEBottleneckConfig,
    L2BottleneckConfig,
    WassersteinBottleneckConfig,
    FSQBottleneckConfig,
    DitheredFSQBottleneckConfig,
] = Field(..., discriminator='type')
```

## Usage Examples

### Simple Bottlenecks
```json
{
    "type": "tanh",
    "config": {"scale": 2.0}
}
```

### Vector Quantization
```json
{
    "type": "rvq",
    "config": {
        "dim": 256,
        "codebook_size": 2048,
        "num_quantizers": 12
    }
}
```

### DAC Quantization
```json
{
    "type": "dac_rvq",
    "config": {
        "n_codebooks": 16,
        "codebook_size": 1024,
        "quantize_on_decode": false
    }
}
```

## Migration Notes

- Current implementation merges default parameters with config using `dict.update()`
- Pydantic migration will provide:
  - Proper validation of all parameters
  - Type checking for numeric values
  - Automatic application of defaults
  - Better error messages for invalid configurations
  - Documentation of all parameters 