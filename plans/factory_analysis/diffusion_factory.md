# Diffusion Factory Analysis

## Overview

The diffusion factory in `stable_audio_tools/models/diffusion.py` creates diffusion models for unconditional and conditional audio generation. These factories build complete diffusion models with proper conditioning and pretransform handling.

## Factory Functions

### `create_diffusion_uncond_from_config(config)`

**Purpose**: Creates unconditional diffusion models from configuration
**Location**: `stable_audio_tools/models/diffusion.py:578`

### `create_diffusion_cond_from_config(config)`

**Purpose**: Creates conditional diffusion models from configuration
**Location**: `stable_audio_tools/models/diffusion.py:628`

## Configuration Schema

### Common Top-Level Structure
```python
{
    "model": dict,  # Required - model configuration
    "model_type": str,  # Required for conditional - model type identifier
    "sample_size": int,  # Required for unconditional - sample size
    "sample_rate": int,  # Required - audio sample rate
}
```

### Common Top-Level Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `model` | `dict` | Model configuration | **Required** | Must contain required model fields |
| `sample_rate` | `int` | Audio sample rate | **Required** | Must be positive integer |
| `sample_size` | `int` | Sample size (uncond only) | **Required** | Must be positive integer |
| `model_type` | `str` | Model type (cond only) | **Required** | Must be supported type |

## Unconditional Diffusion Configuration

### `create_diffusion_uncond_from_config` Schema

```python
{
    "model": {
        "type": str,  # Required - diffusion model type
        "config": dict,  # Optional - model-specific configuration
        "pretransform": dict,  # Optional - pretransform configuration
    },
    "sample_size": int,  # Required - sample size
    "sample_rate": int,  # Required - audio sample rate
}
```

### Model Section Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `type` | `str` | Diffusion model type | **Required** | Must be supported type |
| `config` | `dict` | Model-specific configuration | `{}` | Structure depends on type |
| `pretransform` | `dict` | Pretransform configuration | `None` | Optional - see pretransform factory |

### Supported Unconditional Types

#### 1. `DAU1d` - Diffusion Attention UNet 1D
- **Class**: `DiffusionAttnUnet1D`
- **Purpose**: 1D UNet with attention for diffusion
- **Wrapper**: `DiffusionModelWrapper`

#### 2. `adp_uncond_1d` - ADP Unconditional 1D
- **Class**: `UNet1DUncondWrapper`
- **Purpose**: 1D UNet for unconditional generation
- **Wrapper**: `DiffusionModelWrapper`

#### 3. `dit` - DiT Unconditional
- **Class**: `DiTUncondWrapper`
- **Purpose**: Transformer-based unconditional diffusion
- **Wrapper**: `DiffusionModelWrapper`

## Conditional Diffusion Configuration

### `create_diffusion_cond_from_config` Schema

```python
{
    "model": {
        "diffusion": dict,  # Required - diffusion configuration
        "io_channels": int,  # Required - I/O channels
        "pretransform": dict,  # Optional - pretransform configuration
        "conditioning": dict,  # Optional - conditioning configuration
    },
    "model_type": str,  # Required - model type
    "sample_rate": int,  # Required - audio sample rate
}
```

### Diffusion Configuration Structure

```python
{
    "diffusion": {
        "type": str,  # Required - diffusion model type
        "config": dict,  # Required - diffusion model configuration
        "diffusion_objective": str,  # Optional - diffusion objective
        "cross_attention_cond_ids": list,  # Optional - cross attention conditioning IDs
        "global_cond_ids": list,  # Optional - global conditioning IDs
        "input_concat_ids": list,  # Optional - input concatenation IDs
        "prepend_cond_ids": list,  # Optional - prepend conditioning IDs
        "distribution_shift_options": dict,  # Optional - distribution shift options
    }
}
```

### Diffusion Section Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `type` | `str` | Diffusion model type | **Required** | Must be supported type |
| `config` | `dict` | Model-specific configuration | **Required** | Structure depends on type |
| `diffusion_objective` | `str` | Diffusion objective | `"v"` | Must be supported objective |
| `cross_attention_cond_ids` | `list` | Cross attention conditioning IDs | `[]` | List of strings |
| `global_cond_ids` | `list` | Global conditioning IDs | `[]` | List of strings |
| `input_concat_ids` | `list` | Input concatenation IDs | `[]` | List of strings |
| `prepend_cond_ids` | `list` | Prepend conditioning IDs | `[]` | List of strings |
| `distribution_shift_options` | `dict` | Distribution shift options | `None` | Optional configuration |

### Supported Conditional Types

#### 1. `adp_cfg_1d` - ADP CFG 1D
- **Class**: `UNetCFG1DWrapper`
- **Purpose**: 1D UNet with classifier-free guidance
- **Wrapper**: `ConditionedDiffusionModelWrapper`

#### 2. `adp_1d` - ADP 1D
- **Class**: `UNet1DCondWrapper`
- **Purpose**: 1D UNet with conditioning
- **Wrapper**: `ConditionedDiffusionModelWrapper`

#### 3. `dit` - DiT Conditional
- **Class**: `DiTWrapper`
- **Purpose**: Transformer-based conditional diffusion
- **Wrapper**: `ConditionedDiffusionModelWrapper`

## Model Type Mapping

| Model Type | Factory | Wrapper Class | Purpose |
|------------|---------|---------------|---------|
| `diffusion_uncond` | `create_diffusion_uncond_from_config` | `DiffusionModelWrapper` | Unconditional generation |
| `diffusion_cond` | `create_diffusion_cond_from_config` | `ConditionedDiffusionModelWrapper` | Conditional generation |
| `diffusion_cond_inpaint` | `create_diffusion_cond_from_config` | `ConditionedDiffusionModelWrapper` | Conditional inpainting |

## Error Handling

### Unconditional Diffusion Errors
- **Missing model type**: `AssertionError` - "Must specify model type in config"
- **Missing sample_size**: `AssertionError` - "Must specify sample size in config"
- **Missing sample_rate**: `AssertionError` - "Must specify sample rate in config"
- **Unknown model type**: `NotImplementedError` - "Unknown model type: {model_type}"

### Conditional Diffusion Errors
- **Missing diffusion config**: `AssertionError` - "Must specify diffusion config"
- **Missing diffusion model type**: `AssertionError` - "Must specify diffusion model type"
- **Missing diffusion model config**: `AssertionError` - "Must specify diffusion model config"
- **Missing io_channels**: `AssertionError` - "Must specify io_channels in model config"
- **Missing sample_rate**: `AssertionError` - "Must specify sample_rate in config"

## Special Processing

### Minimum Input Length Calculation

1. **Base Length**: 
   - With pretransform: `pretransform.downsampling_ratio`
   - Without pretransform: `1`

2. **Model-Specific Multipliers**:
   - `adp_cfg_1d` / `adp_1d`: `np.prod(config["factors"])`
   - `dit`: `diffusion_model.model.patch_size`
   - `DAU1d`: No additional multiplier

### Conditioning Setup

1. **Multi-Conditioner**: Created from `conditioning` configuration if provided
2. **Conditioning IDs**: Extracted from diffusion configuration
3. **Pretransform**: Passed to conditioner if available

## Return Types

### Unconditional Models
- **Class**: `DiffusionModelWrapper`
- **Module**: `stable_audio_tools.models.diffusion`

### Conditional Models
- **Class**: `ConditionedDiffusionModelWrapper`
- **Module**: `stable_audio_tools.models.diffusion`

## Pydantic Schema Design

### Base Configuration

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Literal
from enum import Enum

class UnconditionalDiffusionType(str, Enum):
    DAU1D = "DAU1d"
    ADP_UNCOND_1D = "adp_uncond_1d"
    DIT = "dit"

class ConditionalDiffusionType(str, Enum):
    ADP_CFG_1D = "adp_cfg_1d"
    ADP_1D = "adp_1d"
    DIT = "dit"

class DiffusionObjective(str, Enum):
    V = "v"
    RECTIFIED_FLOW = "rectified_flow"
    RF_DENOISER = "rf_denoiser"

class UnconditionalDiffusionModelConfig(BaseModel):
    """Configuration for unconditional diffusion model"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: UnconditionalDiffusionType = Field(..., description="Diffusion model type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model-specific configuration")
    pretransform: Optional[Dict[str, Any]] = Field(None, description="Pretransform configuration")

class ConditionalDiffusionConfig(BaseModel):
    """Configuration for conditional diffusion model"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: ConditionalDiffusionType = Field(..., description="Diffusion model type")
    config: Dict[str, Any] = Field(..., description="Model-specific configuration")
    diffusion_objective: DiffusionObjective = Field(DiffusionObjective.V, description="Diffusion objective")
    cross_attention_cond_ids: List[str] = Field(default_factory=list, description="Cross attention conditioning IDs")
    global_cond_ids: List[str] = Field(default_factory=list, description="Global conditioning IDs")
    input_concat_ids: List[str] = Field(default_factory=list, description="Input concatenation IDs")
    prepend_cond_ids: List[str] = Field(default_factory=list, description="Prepend conditioning IDs")
    distribution_shift_options: Optional[Dict[str, Any]] = Field(None, description="Distribution shift options")

class ConditionalDiffusionModelConfig(BaseModel):
    """Configuration for conditional diffusion model"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    diffusion: ConditionalDiffusionConfig = Field(..., description="Diffusion configuration")
    io_channels: int = Field(..., description="I/O channels", gt=0)
    pretransform: Optional[Dict[str, Any]] = Field(None, description="Pretransform configuration")
    conditioning: Optional[Dict[str, Any]] = Field(None, description="Conditioning configuration")

class UnconditionalDiffusionConfigContainer(BaseModel):
    """Container for unconditional diffusion configuration"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    model: UnconditionalDiffusionModelConfig = Field(..., description="Model configuration")
    sample_size: int = Field(..., description="Sample size", gt=0)
    sample_rate: int = Field(..., description="Audio sample rate", gt=0)

class ConditionalDiffusionConfigContainer(BaseModel):
    """Container for conditional diffusion configuration"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    model: ConditionalDiffusionModelConfig = Field(..., description="Model configuration")
    model_type: Literal["diffusion_cond", "diffusion_cond_inpaint"] = Field(..., description="Model type")
    sample_rate: int = Field(..., description="Audio sample rate", gt=0)
```

## Usage Examples

### Unconditional Diffusion
```json
{
    "model": {
        "type": "DAU1d",
        "config": {
            "io_channels": 2,
            "depth": 14,
            "channels": [128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        },
        "pretransform": {
            "type": "autoencoder",
            "config": {
                "autoencoder_ckpt": "/path/to/autoencoder.ckpt"
            }
        }
    },
    "sample_size": 1048576,
    "sample_rate": 48000
}
```

### Conditional Diffusion
```json
{
    "model": {
        "diffusion": {
            "type": "dit",
            "config": {
                "io_channels": 128,
                "patch_size": 1,
                "depth": 24,
                "num_heads": 16,
                "transformer_type": "continuous_transformer"
            },
            "diffusion_objective": "v",
            "cross_attention_cond_ids": ["prompt"],
            "global_cond_ids": ["seconds_start", "seconds_total"]
        },
        "io_channels": 128,
        "conditioning": {
            "configs": [
                {
                    "id": "prompt",
                    "type": "t5",
                    "config": {
                        "t5_model_name": "google/flan-t5-large"
                    }
                }
            ]
        }
    },
    "model_type": "diffusion_cond",
    "sample_rate": 44100
}
```

### Minimal Conditional Configuration
```json
{
    "model": {
        "diffusion": {
            "type": "adp_1d",
            "config": {
                "in_channels": 128,
                "channels": 256,
                "factors": [2, 2, 2, 2]
            }
        },
        "io_channels": 128
    },
    "model_type": "diffusion_cond",
    "sample_rate": 22050
}
```

## Dependencies

The diffusion factories depend on:
- `stable_audio_tools.models.diffusion` - Diffusion model classes and wrappers
- `stable_audio_tools.models.factory` - Pretransform factory
- `stable_audio_tools.models.conditioners` - Multi-conditioner factory
- `stable_audio_tools.models.adp` - ADP model implementations
- `stable_audio_tools.models.dit` - DiT model implementations

## Migration Notes

### Current Implementation Issues
- Complex minimum input length calculation logic
- Inconsistent error handling between unconditional and conditional
- Hard-coded model type strings throughout code
- No validation of conditioning ID consistency

### Pydantic Migration Benefits
- Proper validation of all required fields
- Type-safe enums for model types and objectives
- Consistent error messages across all configurations
- Validation of conditioning ID relationships
- Documentation of all parameters and their purposes
- Elimination of runtime assertion errors through compile-time validation 