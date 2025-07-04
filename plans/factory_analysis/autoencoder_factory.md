# Autoencoder Factory Analysis

## Overview

The autoencoder factory in `stable_audio_tools/models/autoencoders.py` creates high-level autoencoder modules by combining encoder, decoder, bottleneck, and pretransform components. These factories build complete audio autoencoder models.

## Factory Functions

### `create_autoencoder_from_config(config)`

**Purpose**: Creates standard audio autoencoder from configuration
**Location**: `stable_audio_tools/models/autoencoders.py:866`

### `create_diffAE_from_config(config)`

**Purpose**: Creates diffusion autoencoder from configuration
**Location**: `stable_audio_tools/models/autoencoders.py:911`

### Missing Factory Function

**Issue**: The main factory references `create_hyperencoder_from_config` but this function does not exist in the codebase.
**Location**: `stable_audio_tools/models/factory.py:11`

## Configuration Schema

### Common Structure (Top-Level Config)
```python
{
    "model": dict,  # Required - model-specific configuration
    "sample_rate": int,  # Required - audio sample rate
    # Additional fields depend on model type
}
```

### Common Top-Level Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `model` | `dict` | Model configuration | **Required** | Must contain required model fields |
| `sample_rate` | `int` | Audio sample rate | **Required** | Must be positive integer |

## Standard Autoencoder Configuration

### `create_autoencoder_from_config` Schema

```python
{
    "model": {
        "encoder": dict,  # Required - encoder configuration
        "decoder": dict,  # Required - decoder configuration
        "latent_dim": int,  # Required - latent dimension
        "downsampling_ratio": int,  # Required - downsampling ratio
        "io_channels": int,  # Required - I/O channels
        "in_channels": int,  # Optional - input channels
        "out_channels": int,  # Optional - output channels
        "bottleneck": dict,  # Optional - bottleneck configuration
        "pretransform": dict,  # Optional - pretransform configuration
    },
    "sample_rate": int,  # Required - audio sample rate
}
```

### Model Section Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `encoder` | `dict` | Encoder configuration | **Required** | Must be valid encoder config |
| `decoder` | `dict` | Decoder configuration | **Required** | Must be valid decoder config |
| `latent_dim` | `int` | Latent dimension | **Required** | Must be positive integer |
| `downsampling_ratio` | `int` | Downsampling ratio | **Required** | Must be positive integer |
| `io_channels` | `int` | I/O channels | **Required** | Must be positive integer |
| `in_channels` | `int` | Input channels | `None` | Optional override |
| `out_channels` | `int` | Output channels | `None` | Optional override |
| `bottleneck` | `dict` | Bottleneck configuration | `None` | Optional - see bottleneck factory |
| `pretransform` | `dict` | Pretransform configuration | `None` | Optional - see pretransform factory |

### Special Processing

1. **Soft Clipping**: Extracted from decoder configuration
   - Path: `config["model"]["decoder"]["soft_clip"]`
   - Default: `False`

2. **Encoder/Decoder**: Created using respective factory functions
   - Uses `create_encoder_from_config()`
   - Uses `create_decoder_from_config()`

3. **Bottleneck**: Optional processing component
   - Uses `create_bottleneck_from_config()` if provided

4. **Pretransform**: Optional preprocessing component
   - Uses `create_pretransform_from_config()` if provided

## Diffusion Autoencoder Configuration

### `create_diffAE_from_config` Schema

```python
{
    "model": {
        "encoder": dict,  # Optional - encoder configuration
        "decoder": dict,  # Optional - decoder configuration
        "diffusion": dict,  # Required - diffusion configuration
        "latent_dim": int,  # Required - latent dimension
        "downsampling_ratio": int,  # Required - downsampling ratio
        "io_channels": int,  # Required - I/O channels
        "bottleneck": dict,  # Optional - bottleneck configuration
        "pretransform": dict,  # Optional - pretransform configuration
    },
    "sample_rate": int,  # Required - audio sample rate
}
```

### Diffusion Configuration

```python
{
    "diffusion": {
        "type": str,  # Required - diffusion model type
        "config": dict,  # Required - diffusion-specific configuration
    }
}
```

### Supported Diffusion Types

#### 1. `DAU1d` - DAU1D Conditional Wrapper
- **Class**: `DAU1DCondWrapper`
- **Purpose**: 1D diffusion with conditional generation
- **Downsampling**: Computed from `config["strides"]`

#### 2. `adp_1d` - ADP 1D UNet
- **Class**: `UNet1DCondWrapper`
- **Purpose**: 1D UNet diffusion model
- **Downsampling**: Computed from `config["factors"]`

#### 3. `dit` - DiT (Diffusion in Time)
- **Class**: `DiTWrapper`
- **Purpose**: Transformer-based diffusion
- **Downsampling**: Fixed to `1`

| Type | Class | Downsampling Source | Purpose |
|------|-------|-------------------|---------|
| `DAU1d` | `DAU1DCondWrapper` | `config["strides"]` | 1D conditional diffusion |
| `adp_1d` | `UNet1DCondWrapper` | `config["factors"]` | 1D UNet diffusion |
| `dit` | `DiTWrapper` | `1` (fixed) | Transformer diffusion |

### Diffusion-Specific Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `encoder` | `dict` | Encoder configuration | `None` | Optional - can be None |
| `decoder` | `dict` | Decoder configuration | `None` | Optional - can be None |
| `diffusion` | `dict` | Diffusion configuration | **Required** | Must specify type and config |

## Error Handling

### Standard Autoencoder Errors
- **Missing latent_dim**: `AssertionError` - "latent_dim must be specified in model config"
- **Missing downsampling_ratio**: `AssertionError` - "downsampling_ratio must be specified in model config"
- **Missing io_channels**: `AssertionError` - "io_channels must be specified in model config"
- **Missing sample_rate**: `AssertionError` - "sample_rate must be specified in model config"

### Diffusion Autoencoder Errors
- **Missing diffusion type**: `KeyError` when accessing `diffusion["type"]`
- **Unknown diffusion type**: No explicit error - will fail during class instantiation

## Return Types

### Standard Autoencoder
- **Class**: `AudioAutoencoder`
- **Module**: `stable_audio_tools.models.autoencoders`

### Diffusion Autoencoder
- **Class**: `DiffusionAutoencoder`
- **Module**: `stable_audio_tools.models.autoencoders`
- **Inheritance**: Extends `AudioAutoencoder`

## Pydantic Schema Design

### Base Configuration

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Union
from enum import Enum

class DiffusionType(str, Enum):
    DAU1D = "DAU1d"
    ADP_1D = "adp_1d"
    DIT = "dit"

class AutoencoderConfig(BaseModel):
    """Configuration for standard autoencoder"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    encoder: Dict[str, Any] = Field(..., description="Encoder configuration")
    decoder: Dict[str, Any] = Field(..., description="Decoder configuration")
    latent_dim: int = Field(..., description="Latent dimension", gt=0)
    downsampling_ratio: int = Field(..., description="Downsampling ratio", gt=0)
    io_channels: int = Field(..., description="I/O channels", gt=0)
    in_channels: Optional[int] = Field(None, description="Input channels override")
    out_channels: Optional[int] = Field(None, description="Output channels override")
    bottleneck: Optional[Dict[str, Any]] = Field(None, description="Bottleneck configuration")
    pretransform: Optional[Dict[str, Any]] = Field(None, description="Pretransform configuration")
    
    @validator('in_channels', 'out_channels')
    def validate_channels(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Channel count must be positive")
        return v

class DiffusionConfig(BaseModel):
    """Configuration for diffusion model"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: DiffusionType = Field(..., description="Diffusion model type")
    config: Dict[str, Any] = Field(..., description="Diffusion-specific configuration")

class DiffusionAutoencoderConfig(BaseModel):
    """Configuration for diffusion autoencoder"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    encoder: Optional[Dict[str, Any]] = Field(None, description="Encoder configuration")
    decoder: Optional[Dict[str, Any]] = Field(None, description="Decoder configuration")
    diffusion: DiffusionConfig = Field(..., description="Diffusion configuration")
    latent_dim: int = Field(..., description="Latent dimension", gt=0)
    downsampling_ratio: int = Field(..., description="Downsampling ratio", gt=0)
    io_channels: int = Field(..., description="I/O channels", gt=0)
    bottleneck: Optional[Dict[str, Any]] = Field(None, description="Bottleneck configuration")
    pretransform: Optional[Dict[str, Any]] = Field(None, description="Pretransform configuration")

class BaseAutoencoderConfigContainer(BaseModel):
    """Base container for autoencoder configurations"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    sample_rate: int = Field(..., description="Audio sample rate", gt=0)
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        if v <= 0:
            raise ValueError("Sample rate must be positive")
        return v

class StandardAutoencoderConfigContainer(BaseAutoencoderConfigContainer):
    """Container for standard autoencoder configuration"""
    model: AutoencoderConfig = Field(..., description="Autoencoder model configuration")

class DiffusionAutoencoderConfigContainer(BaseAutoencoderConfigContainer):
    """Container for diffusion autoencoder configuration"""
    model: DiffusionAutoencoderConfig = Field(..., description="Diffusion autoencoder configuration")
```

## Usage Examples

### Standard Autoencoder
```json
{
    "model": {
        "encoder": {
            "type": "oobleck",
            "config": {
                "in_channels": 2,
                "latent_dim": 128,
                "channels": 256
            }
        },
        "decoder": {
            "type": "oobleck",
            "config": {
                "out_channels": 2,
                "latent_dim": 128,
                "channels": 256,
                "final_tanh": true
            },
            "soft_clip": false
        },
        "latent_dim": 128,
        "downsampling_ratio": 2048,
        "io_channels": 2,
        "bottleneck": {
            "type": "vae",
            "config": {}
        }
    },
    "sample_rate": 48000
}
```

### Diffusion Autoencoder
```json
{
    "model": {
        "encoder": {
            "type": "oobleck",
            "config": {
                "in_channels": 2,
                "latent_dim": 64
            }
        },
        "decoder": {
            "type": "oobleck",
            "config": {
                "out_channels": 2,
                "latent_dim": 64
            }
        },
        "diffusion": {
            "type": "dit",
            "config": {
                "transformer_depth": 12,
                "num_heads": 16
            }
        },
        "latent_dim": 64,
        "downsampling_ratio": 1024,
        "io_channels": 2
    },
    "sample_rate": 44100
}
```

### Minimal Configuration
```json
{
    "model": {
        "encoder": {
            "type": "oobleck",
            "config": {}
        },
        "decoder": {
            "type": "oobleck",
            "config": {}
        },
        "latent_dim": 32,
        "downsampling_ratio": 512,
        "io_channels": 2
    },
    "sample_rate": 22050
}
```

## Dependencies

The autoencoder factories depend on:
- `stable_audio_tools.models.autoencoders` - Autoencoder classes
- `stable_audio_tools.models.factory` - Pretransform and bottleneck factories
- Encoder and decoder factories (same module)
- Diffusion model classes (`DAU1DCondWrapper`, `UNet1DCondWrapper`, `DiTWrapper`)

## Migration Notes

### Missing Function Issue
- **Problem**: `create_hyperencoder_from_config` is referenced but not implemented
- **Solution Options**:
  1. Implement missing function
  2. Remove hyperencoder model type
  3. Alias to existing autoencoder factory

### Current Implementation Issues
- Soft clipping extraction assumes decoder config structure
- No validation for required diffusion config fields
- Diffusion downsampling ratio calculation logic is complex and error-prone

### Pydantic Migration Benefits
- Proper validation of all required fields
- Type checking for all configuration parameters
- Better error messages for missing or invalid configurations
- Unified configuration structure
- Documentation of all parameters and their purposes
- Elimination of runtime assertion errors through compile-time validation 