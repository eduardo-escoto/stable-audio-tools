# Encoder/Decoder Factory Analysis

## Overview

The encoder and decoder factories in `stable_audio_tools/models/autoencoders.py` create encoder and decoder modules for autoencoder architectures. These are fundamental building blocks used by higher-level autoencoder factories.

## Factory Functions

### `create_encoder_from_config(encoder_config)`

**Purpose**: Creates encoder modules from configuration
**Location**: `stable_audio_tools/models/autoencoders.py:782`

### `create_decoder_from_config(decoder_config)`

**Purpose**: Creates decoder modules from configuration  
**Location**: `stable_audio_tools/models/autoencoders.py:826`

## Configuration Schema

### Common Structure
```python
{
    "type": str,  # Required - encoder/decoder type identifier
    "config": dict,  # Required - type-specific parameters
    "requires_grad": bool,  # Optional - whether parameters require gradients
}
```

### Common Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `type` | `str` | Encoder/decoder type identifier | **Required** | Must be one of supported types |
| `config` | `dict` | Type-specific configuration | **Required** | Structure depends on type |
| `requires_grad` | `bool` | Whether parameters require gradients | `True` | Boolean |

## Supported Encoder Types

### 1. `oobleck` - Oobleck Encoder
- **Class**: `OobleckEncoder`
- **Purpose**: Custom encoder architecture
- **Module**: `stable_audio_tools.models.autoencoders`

#### Configuration
```python
{
    "type": "oobleck",
    "config": {
        "in_channels": int,  # Optional, default: 2
        "channels": int,  # Optional, default: 128
        "latent_dim": int,  # Optional, default: 32
        "c_mults": list,  # Optional, default: [1, 2, 4, 8]
        "strides": list,  # Optional, default: [2, 4, 8, 8]
        "use_snake": bool,  # Optional, default: False
        "antialias_activation": bool,  # Optional, default: False
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `in_channels` | `int` | `2` | Number of input channels | Must be positive |
| `channels` | `int` | `128` | Base number of channels | Must be positive |
| `latent_dim` | `int` | `32` | Latent dimension | Must be positive |
| `c_mults` | `list` | `[1, 2, 4, 8]` | Channel multipliers | List of positive integers |
| `strides` | `list` | `[2, 4, 8, 8]` | Stride values | List of positive integers |
| `use_snake` | `bool` | `False` | Use Snake activation | Boolean |
| `antialias_activation` | `bool` | `False` | Use antialiased activation | Boolean |

### 2. `seanet` - SEANet Encoder
- **Class**: `SEANetEncoder` (from encodec)
- **Purpose**: Encodec's SEANet encoder architecture
- **Module**: `encodec.modules`

#### Configuration
```python
{
    "type": "seanet",
    "config": {
        "ratios": list,  # Optional, default: [2, 2, 2, 2, 2] (reversed internally)
        # Additional SEANet parameters
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `ratios` | `list` | `[2, 2, 2, 2, 2]` | Downsampling ratios (reversed internally) | List of positive integers |

**Note**: SEANet encoder expects strides in reverse order, so ratios are automatically reversed.

### 3. `dac` - DAC Encoder
- **Class**: `DACEncoderWrapper`
- **Purpose**: Wrapper for DAC encoder
- **Module**: `stable_audio_tools.models.autoencoders`

#### Configuration
```python
{
    "type": "dac",
    "config": {
        "in_channels": int,  # Optional, default: 1
        # Additional DAC encoder parameters
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `in_channels` | `int` | `1` | Number of input channels | Must be positive |

### 4. `local_attn` - Local Attention Encoder
- **Class**: `TransformerEncoder1D`
- **Purpose**: Transformer-based encoder with local attention
- **Module**: `stable_audio_tools.models.local_attention`

#### Configuration
```python
{
    "type": "local_attn",
    "config": {
        # TransformerEncoder1D parameters
    }
}
```

**Note**: All parameters in `config` are passed directly to `TransformerEncoder1D` constructor.

### 5. `taae` - TAAE Encoder
- **Class**: `TAAEEncoder`
- **Purpose**: Transformer Audio Autoencoder encoder
- **Module**: `stable_audio_tools.models.autoencoders`

#### Configuration
```python
{
    "type": "taae",
    "config": {
        "in_channels": int,  # Optional, default: 2
        "channels": int,  # Optional, default: 128
        "latent_dim": int,  # Optional, default: 32
        "c_mults": list,  # Optional, default: [1, 2, 4, 8]
        "strides": list,  # Optional, default: [2, 4, 8, 8]
        "transformer_depths": list,  # Optional, default: [3, 3, 3, 3]
        "use_snake": bool,  # Optional, default: False
        "sliding_window": list,  # Optional, default: [63, 64]
        "checkpointing": bool,  # Optional, default: False
        "conformer": bool,  # Optional, default: False
        "layer_scale": bool,  # Optional, default: True
        "use_dilated_conv": bool,  # Optional, default: False
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `in_channels` | `int` | `2` | Number of input channels | Must be positive |
| `channels` | `int` | `128` | Base number of channels | Must be positive |
| `latent_dim` | `int` | `32` | Latent dimension | Must be positive |
| `c_mults` | `list` | `[1, 2, 4, 8]` | Channel multipliers | List of positive integers |
| `strides` | `list` | `[2, 4, 8, 8]` | Stride values | List of positive integers |
| `transformer_depths` | `list` | `[3, 3, 3, 3]` | Transformer depths per layer | List of positive integers |
| `use_snake` | `bool` | `False` | Use Snake activation | Boolean |
| `sliding_window` | `list` | `[63, 64]` | Sliding window sizes | List of positive integers |
| `checkpointing` | `bool` | `False` | Use gradient checkpointing | Boolean |
| `conformer` | `bool` | `False` | Use Conformer blocks | Boolean |
| `layer_scale` | `bool` | `True` | Use layer scaling | Boolean |
| `use_dilated_conv` | `bool` | `False` | Use dilated convolutions | Boolean |

## Supported Decoder Types

### 1. `oobleck` - Oobleck Decoder
- **Class**: `OobleckDecoder`
- **Purpose**: Custom decoder architecture (mirrors Oobleck encoder)

#### Configuration
```python
{
    "type": "oobleck",
    "config": {
        "out_channels": int,  # Optional, default: 2
        "channels": int,  # Optional, default: 128
        "latent_dim": int,  # Optional, default: 32
        "c_mults": list,  # Optional, default: [1, 2, 4, 8]
        "strides": list,  # Optional, default: [2, 4, 8, 8]
        "use_snake": bool,  # Optional, default: False
        "antialias_activation": bool,  # Optional, default: False
        "use_nearest_upsample": bool,  # Optional, default: False
        "final_tanh": bool,  # Optional, default: True
        "soft_clip": bool,  # Passed from parent config
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `out_channels` | `int` | `2` | Number of output channels | Must be positive |
| `final_tanh` | `bool` | `True` | Apply tanh activation at output | Boolean |
| `use_nearest_upsample` | `bool` | `False` | Use nearest upsampling | Boolean |
| `soft_clip` | `bool` | `False` | Apply soft clipping | Boolean |

### 2. `seanet` - SEANet Decoder
- **Class**: `SEANetDecoder` (from encodec)
- **Purpose**: Encodec's SEANet decoder architecture

#### Configuration
```python
{
    "type": "seanet",
    "config": {
        # SEANet decoder parameters
        "soft_clip": bool,  # Passed from parent config
    }
}
```

### 3. `dac` - DAC Decoder
- **Class**: `DACDecoderWrapper`
- **Purpose**: Wrapper for DAC decoder

#### Configuration
```python
{
    "type": "dac",
    "config": {
        "latent_dim": int,  # Required
        "out_channels": int,  # Optional, default: 1
        # Additional DAC decoder parameters
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `latent_dim` | `int` | **Required** | Latent dimension | Must be positive |
| `out_channels` | `int` | `1` | Number of output channels | Must be positive |

### 4. `local_attn` - Local Attention Decoder
- **Class**: `TransformerDecoder1D`
- **Purpose**: Transformer-based decoder with local attention

#### Configuration
```python
{
    "type": "local_attn",
    "config": {
        # TransformerDecoder1D parameters
    }
}
```

### 5. `taae` - TAAE Decoder
- **Class**: `TAAEDecoder`
- **Purpose**: Transformer Audio Autoencoder decoder

#### Configuration
```python
{
    "type": "taae",
    "config": {
        "out_channels": int,  # Optional, default: 2
        "channels": int,  # Optional, default: 128
        "latent_dim": int,  # Optional, default: 32
        "c_mults": list,  # Optional, default: [1, 2, 4, 8]
        "strides": list,  # Optional, default: [2, 4, 8, 8]
        "transformer_depths": list,  # Optional, default: [3, 3, 3, 3]
        "use_snake": bool,  # Optional, default: False
        "sliding_window": list,  # Optional, default: [63, 64]
        "checkpointing": bool,  # Optional, default: False
        "conformer": bool,  # Optional, default: False
        "layer_scale": bool,  # Optional, default: True
        "use_dilated_conv": bool,  # Optional, default: False
    }
}
```

## Error Handling

### Encoder Factory
- **Missing type**: Raises `AssertionError` with message "Encoder type must be specified"
- **Unknown type**: Raises `ValueError` with message "Unknown encoder type {encoder_type}"

### Decoder Factory
- **Missing type**: Raises `AssertionError` with message "Decoder type must be specified"
- **Unknown type**: Raises `ValueError` with message "Unknown decoder type {decoder_type}"

## Post-Processing

Both factories apply gradient control:
- Reads `requires_grad` from configuration (defaults to `True`)
- If `requires_grad` is `False`, sets all parameters to not require gradients

## Pydantic Schema Design

### Base Encoder/Decoder Config

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum

class EncoderType(str, Enum):
    OOBLECK = "oobleck"
    SEANET = "seanet"
    DAC = "dac"
    LOCAL_ATTN = "local_attn"
    TAAE = "taae"

class DecoderType(str, Enum):
    OOBLECK = "oobleck"
    SEANET = "seanet"
    DAC = "dac"
    LOCAL_ATTN = "local_attn"
    TAAE = "taae"

class BaseEncoderConfig(BaseModel):
    """Base configuration for encoder modules"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: EncoderType = Field(..., description="Type of encoder")
    config: Dict[str, Any] = Field(..., description="Type-specific configuration")
    requires_grad: bool = Field(True, description="Whether parameters require gradients")

class BaseDecoderConfig(BaseModel):
    """Base configuration for decoder modules"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: DecoderType = Field(..., description="Type of decoder")
    config: Dict[str, Any] = Field(..., description="Type-specific configuration")
    requires_grad: bool = Field(True, description="Whether parameters require gradients")
```

### Specific Encoder Configs

```python
class OobleckEncoderConfig(BaseEncoderConfig):
    type: Literal[EncoderType.OOBLECK] = EncoderType.OOBLECK
    config: Dict[str, Any] = Field(default_factory=lambda: {
        "in_channels": 2,
        "channels": 128,
        "latent_dim": 32,
        "c_mults": [1, 2, 4, 8],
        "strides": [2, 4, 8, 8],
        "use_snake": False,
        "antialias_activation": False,
    })
    
    @validator('config')
    def validate_oobleck_config(cls, v):
        if 'in_channels' in v and v['in_channels'] <= 0:
            raise ValueError("in_channels must be positive")
        if 'channels' in v and v['channels'] <= 0:
            raise ValueError("channels must be positive")
        if 'c_mults' in v and not all(m > 0 for m in v['c_mults']):
            raise ValueError("c_mults must be positive integers")
        if 'strides' in v and not all(s > 0 for s in v['strides']):
            raise ValueError("strides must be positive integers")
        return v

class TAAEEncoderConfig(BaseEncoderConfig):
    type: Literal[EncoderType.TAAE] = EncoderType.TAAE
    config: Dict[str, Any] = Field(default_factory=lambda: {
        "in_channels": 2,
        "channels": 128,
        "latent_dim": 32,
        "c_mults": [1, 2, 4, 8],
        "strides": [2, 4, 8, 8],
        "transformer_depths": [3, 3, 3, 3],
        "use_snake": False,
        "sliding_window": [63, 64],
        "checkpointing": False,
        "conformer": False,
        "layer_scale": True,
        "use_dilated_conv": False,
    })
    
    @validator('config')
    def validate_taae_config(cls, v):
        # Validate transformer_depths
        if 'transformer_depths' in v:
            if not all(d > 0 for d in v['transformer_depths']):
                raise ValueError("transformer_depths must be positive integers")
        return v
```

## Usage Examples

### Oobleck Encoder/Decoder Pair
```json
{
    "encoder": {
        "type": "oobleck",
        "config": {
            "in_channels": 2,
            "channels": 256,
            "latent_dim": 64,
            "c_mults": [1, 2, 4, 8, 16],
            "strides": [2, 4, 4, 8, 8]
        }
    },
    "decoder": {
        "type": "oobleck",
        "config": {
            "out_channels": 2,
            "channels": 256,
            "latent_dim": 64,
            "c_mults": [1, 2, 4, 8, 16],
            "strides": [2, 4, 4, 8, 8],
            "final_tanh": true
        }
    }
}
```

### TAAE Configuration
```json
{
    "encoder": {
        "type": "taae",
        "config": {
            "in_channels": 2,
            "transformer_depths": [6, 6, 6, 6],
            "checkpointing": true,
            "conformer": true
        }
    },
    "decoder": {
        "type": "taae",
        "config": {
            "out_channels": 2,
            "transformer_depths": [6, 6, 6, 6],
            "checkpointing": true,
            "conformer": true
        }
    }
}
```

## Dependencies

The encoder/decoder factories depend on:
- `stable_audio_tools.models.autoencoders` - Local encoder/decoder implementations
- `encodec.modules` - SEANet encoder/decoder
- `stable_audio_tools.models.local_attention` - Transformer implementations
- Various external libraries for specialized encoders/decoders

## Migration Notes

- Current implementation uses direct parameter passing to constructors
- SEANet has special handling for reversed ratios
- Soft clipping parameter is passed from parent decoder configuration
- Pydantic migration will provide:
  - Proper validation of all parameters
  - Type checking for all configuration fields
  - Better error messages for invalid configurations
  - Documentation of all parameters
  - Unified configuration structure across encoder types 