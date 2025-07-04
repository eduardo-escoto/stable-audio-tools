# Training Factory Analysis

## Overview

The training factory in `stable_audio_tools/training/factory.py` creates training wrappers and demo callbacks for different model types. These factories provide model-specific training logic and demonstration capabilities.

## Factory Functions

### `create_training_wrapper_from_config(model_config, model)`

**Purpose**: Creates training wrappers from model configuration
**Location**: `stable_audio_tools/training/factory.py:4`

### `create_demo_callback_from_config(model_config, **kwargs)`

**Purpose**: Creates demo callbacks from model configuration
**Location**: `stable_audio_tools/training/factory.py:170`

## Configuration Schema

### Common Structure
```python
{
    "model_type": str,  # Required - model type identifier
    "training": dict,  # Required - training configuration
    "sample_rate": int,  # Required - audio sample rate
    "sample_size": int,  # Required for some models - sample size
}
```

### Common Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `model_type` | `str` | Model type identifier | **Required** | Must be supported type |
| `training` | `dict` | Training configuration | **Required** | Must contain training parameters |
| `sample_rate` | `int` | Audio sample rate | **Required** | Must be positive integer |
| `sample_size` | `int` | Sample size | **Required*** | Must be positive integer |

*Required for some model types

## Supported Model Types

### 1. `autoencoder` - Autoencoder Training
- **Wrapper Class**: `AutoencoderTrainingWrapper`
- **Demo Class**: `AutoencoderDemoCallback`
- **Module**: `stable_audio_tools.training.autoencoders`

#### Training Configuration
```python
{
    "training": {
        "learning_rate": float,  # Optional - learning rate
        "warmup_steps": int,  # Optional - warmup steps
        "encoder_freeze_on_warmup": bool,  # Optional - freeze encoder during warmup
        "use_ema": bool,  # Optional - use exponential moving average
        "force_input_mono": bool,  # Optional - force mono input
        "latent_mask_ratio": float,  # Optional - latent masking ratio
        "loss_configs": dict,  # Optional - loss configurations
        "eval_loss_configs": dict,  # Optional - evaluation loss configurations
        "optimizer_configs": dict,  # Optional - optimizer configurations
        "teacher_model": dict,  # Optional - teacher model configuration
        "teacher_model_ckpt": str,  # Required if teacher_model specified
        "demo": dict,  # Optional - demo configuration
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `learning_rate` | `float` | `None` | Learning rate | Must be positive |
| `warmup_steps` | `int` | `0` | Warmup steps | Must be non-negative |
| `encoder_freeze_on_warmup` | `bool` | `False` | Freeze encoder during warmup | Boolean |
| `use_ema` | `bool` | `False` | Use exponential moving average | Boolean |
| `force_input_mono` | `bool` | `False` | Force mono input | Boolean |
| `latent_mask_ratio` | `float` | `0.0` | Latent masking ratio | Must be between 0 and 1 |
| `teacher_model_ckpt` | `str` | `None` | Teacher model checkpoint | Required if teacher_model specified |

### 2. `diffusion_uncond` - Unconditional Diffusion Training
- **Wrapper Class**: `DiffusionUncondTrainingWrapper`
- **Demo Class**: `DiffusionUncondDemoCallback`
- **Module**: `stable_audio_tools.training.diffusion`

#### Training Configuration
```python
{
    "training": {
        "learning_rate": float,  # Required - learning rate
        "pre_encoded": bool,  # Optional - use pre-encoded data
        "demo": dict,  # Optional - demo configuration
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `learning_rate` | `float` | **Required** | Learning rate | Must be positive |
| `pre_encoded` | `bool` | `False` | Use pre-encoded data | Boolean |

### 3. `diffusion_cond` / `diffusion_cond_inpaint` - Conditional Diffusion Training
- **Wrapper Class**: `DiffusionCondTrainingWrapper` or `ARCTrainingWrapper`
- **Demo Class**: `DiffusionCondDemoCallback`
- **Module**: `stable_audio_tools.training.diffusion` or `stable_audio_tools.training.arc`

#### Standard Conditional Training Configuration
```python
{
    "training": {
        "learning_rate": float,  # Optional - learning rate
        "mask_padding": bool,  # Optional - mask padding
        "mask_padding_dropout": float,  # Optional - mask padding dropout
        "use_ema": bool,  # Optional - use exponential moving average
        "log_loss_info": bool,  # Optional - log loss information
        "optimizer_configs": dict,  # Optional - optimizer configurations
        "pre_encoded": bool,  # Optional - use pre-encoded data
        "cfg_dropout_prob": float,  # Optional - CFG dropout probability
        "timestep_sampler": str,  # Optional - timestep sampler type
        "timestep_sampler_options": dict,  # Optional - timestep sampler options
        "p_one_shot": float,  # Optional - one-shot probability
        "inpainting": dict,  # Optional - inpainting configuration
        "demo": dict,  # Optional - demo configuration
    }
}
```

#### ARC Training Configuration
```python
{
    "training": {
        "arc": {  # Required for ARC training
            "teacher_model": dict,  # Optional - teacher model configuration
            "teacher_model_ckpt": str,  # Required if teacher_model specified
            "use_model_as_teacher": bool,  # Optional - use main model as teacher
            "discriminator_base_model": dict,  # Optional - discriminator model configuration
            "discriminator_base_ckpt": str,  # Optional - discriminator checkpoint
            "use_model_as_discriminator": bool,  # Optional - use main model as discriminator
        },
        "optimizer_configs": dict,  # Optional - optimizer configurations
        "use_ema": bool,  # Optional - use exponential moving average
        "pre_encoded": bool,  # Optional - use pre-encoded data
        "cfg_dropout_prob": float,  # Optional - CFG dropout probability
        "timestep_sampler": str,  # Optional - timestep sampler type
        "clip_grad_norm": float,  # Optional - gradient clipping norm
        "trim_config": dict,  # Optional - trim configuration
        "inpainting": dict,  # Optional - inpainting configuration
    }
}
```

### 4. `diffusion_autoencoder` - Diffusion Autoencoder Training
- **Wrapper Class**: `DiffusionAutoencoderTrainingWrapper`
- **Demo Class**: `DiffusionAutoencoderDemoCallback`
- **Module**: `stable_audio_tools.training.diffusion`

#### Training Configuration
```python
{
    "training": {
        "learning_rate": float,  # Required - learning rate
        "use_reconstruction_loss": bool,  # Optional - use reconstruction loss
        "demo": dict,  # Optional - demo configuration
    }
}
```

### 5. `lm` - Language Model Training
- **Wrapper Class**: `AudioLanguageModelTrainingWrapper`
- **Demo Class**: Not implemented
- **Module**: `stable_audio_tools.training.lm`

#### Training Configuration
```python
{
    "training": {
        "learning_rate": float,  # Optional - learning rate
        "use_ema": bool,  # Optional - use exponential moving average
        "optimizer_configs": dict,  # Optional - optimizer configurations
        "pre_encoded": bool,  # Optional - use pre-encoded data
    }
}
```

## Demo Configuration

### Common Demo Structure
```python
{
    "demo": {
        "demo_every": int,  # Optional - demo frequency
        "demo_steps": int,  # Optional - demo steps (for diffusion)
    }
}
```

### Demo Parameters

| Parameter | Type | Default | Description | Model Types |
|-----------|------|---------|-------------|-------------|
| `demo_every` | `int` | `2000` | Demo frequency in steps | All |
| `demo_steps` | `int` | `250` | Number of diffusion steps | Diffusion models |

## Error Handling

### Training Wrapper Errors
- **Missing model_type**: `AssertionError` - "model_type must be specified in model config"
- **Missing training config**: `AssertionError` - "training config must be specified in model config"
- **Missing teacher_model_ckpt**: `ValueError` - "teacher_model_ckpt must be specified if teacher_model is specified"
- **Unknown model type**: `NotImplementedError` - "Unknown model type: {model_type}"

### Demo Callback Errors
- **Missing model_type**: `AssertionError` - "model_type must be specified in model config"
- **Missing training config**: `AssertionError` - "training config must be specified in model config"

## Special Processing

### EMA (Exponential Moving Average)
1. **Model Duplication**: Creates EMA copy of model if `use_ema` is True
2. **Weight Copying**: Copies weights from main model to EMA copy
3. **Parameter Handling**: Handles serialized parameters for backward compatibility

### Teacher Models
1. **Model Creation**: Creates teacher model from configuration
2. **Checkpoint Loading**: Loads teacher model checkpoint if specified
3. **Evaluation Mode**: Sets teacher model to evaluation mode with no gradients

### Discriminator Models (ARC)
1. **Model Creation**: Creates discriminator from configuration
2. **Checkpoint Loading**: Loads discriminator checkpoint if specified
3. **Base Model Reuse**: Can reuse main model as discriminator base

## Return Types

### Training Wrappers
- **Autoencoder**: `AutoencoderTrainingWrapper`
- **Diffusion**: `DiffusionUncondTrainingWrapper`, `DiffusionCondTrainingWrapper`, `ARCTrainingWrapper`, `DiffusionAutoencoderTrainingWrapper`
- **Language Model**: `AudioLanguageModelTrainingWrapper`

### Demo Callbacks
- **Autoencoder**: `AutoencoderDemoCallback`
- **Diffusion**: `DiffusionUncondDemoCallback`, `DiffusionCondDemoCallback`, `DiffusionAutoencoderDemoCallback`

## Pydantic Schema Design

### Base Configuration

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Union, Literal
from enum import Enum

class ModelType(str, Enum):
    AUTOENCODER = "autoencoder"
    DIFFUSION_UNCOND = "diffusion_uncond"
    DIFFUSION_COND = "diffusion_cond"
    DIFFUSION_COND_INPAINT = "diffusion_cond_inpaint"
    DIFFUSION_AUTOENCODER = "diffusion_autoencoder"
    LM = "lm"

class TimestepSampler(str, Enum):
    UNIFORM = "uniform"
    LOGIT_NORMAL = "logit_normal"
    COSINE = "cosine"

class DemoConfig(BaseModel):
    """Configuration for demo callbacks"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    demo_every: int = Field(2000, description="Demo frequency in steps", gt=0)
    demo_steps: int = Field(250, description="Number of diffusion steps", gt=0)

class BaseTrainingConfig(BaseModel):
    """Base configuration for training"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    learning_rate: Optional[float] = Field(None, description="Learning rate", gt=0)
    optimizer_configs: Optional[Dict[str, Any]] = Field(None, description="Optimizer configurations")
    demo: DemoConfig = Field(default_factory=DemoConfig, description="Demo configuration")

class AutoencoderTrainingConfig(BaseTrainingConfig):
    """Configuration for autoencoder training"""
    warmup_steps: int = Field(0, description="Warmup steps", ge=0)
    encoder_freeze_on_warmup: bool = Field(False, description="Freeze encoder during warmup")
    use_ema: bool = Field(False, description="Use exponential moving average")
    force_input_mono: bool = Field(False, description="Force mono input")
    latent_mask_ratio: float = Field(0.0, description="Latent masking ratio", ge=0, le=1)
    loss_configs: Optional[Dict[str, Any]] = Field(None, description="Loss configurations")
    eval_loss_configs: Optional[Dict[str, Any]] = Field(None, description="Evaluation loss configurations")
    teacher_model: Optional[Dict[str, Any]] = Field(None, description="Teacher model configuration")
    teacher_model_ckpt: Optional[str] = Field(None, description="Teacher model checkpoint")
    
    @validator('teacher_model_ckpt')
    def validate_teacher_checkpoint(cls, v, values):
        if values.get('teacher_model') is not None and v is None:
            raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")
        return v

class DiffusionTrainingConfig(BaseTrainingConfig):
    """Configuration for diffusion training"""
    pre_encoded: bool = Field(False, description="Use pre-encoded data")
    cfg_dropout_prob: float = Field(0.1, description="CFG dropout probability", ge=0, le=1)
    timestep_sampler: TimestepSampler = Field(TimestepSampler.UNIFORM, description="Timestep sampler type")
    timestep_sampler_options: Dict[str, Any] = Field(default_factory=dict, description="Timestep sampler options")

class ConditionalDiffusionTrainingConfig(DiffusionTrainingConfig):
    """Configuration for conditional diffusion training"""
    mask_padding: bool = Field(False, description="Mask padding")
    mask_padding_dropout: float = Field(0.0, description="Mask padding dropout", ge=0, le=1)
    use_ema: bool = Field(True, description="Use exponential moving average")
    log_loss_info: bool = Field(False, description="Log loss information")
    p_one_shot: float = Field(0.0, description="One-shot probability", ge=0, le=1)
    inpainting: Optional[Dict[str, Any]] = Field(None, description="Inpainting configuration")

class ARCConfig(BaseModel):
    """Configuration for ARC training"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    teacher_model: Optional[Dict[str, Any]] = Field(None, description="Teacher model configuration")
    teacher_model_ckpt: Optional[str] = Field(None, description="Teacher model checkpoint")
    use_model_as_teacher: bool = Field(False, description="Use main model as teacher")
    discriminator_base_model: Optional[Dict[str, Any]] = Field(None, description="Discriminator model configuration")
    discriminator_base_ckpt: Optional[str] = Field(None, description="Discriminator checkpoint")
    use_model_as_discriminator: bool = Field(False, description="Use main model as discriminator")

class ARCTrainingConfig(ConditionalDiffusionTrainingConfig):
    """Configuration for ARC training"""
    arc: ARCConfig = Field(..., description="ARC configuration")
    clip_grad_norm: float = Field(0.0, description="Gradient clipping norm", ge=0)
    trim_config: Optional[Dict[str, Any]] = Field(None, description="Trim configuration")
```

## Usage Examples

### Autoencoder Training
```json
{
    "model_type": "autoencoder",
    "training": {
        "learning_rate": 1e-4,
        "warmup_steps": 1000,
        "use_ema": true,
        "latent_mask_ratio": 0.1,
        "loss_configs": {
            "reconstruction": {"weight": 1.0},
            "adversarial": {"weight": 0.1}
        },
        "demo": {
            "demo_every": 2000
        }
    },
    "sample_rate": 48000,
    "sample_size": 1048576
}
```

### Conditional Diffusion Training
```json
{
    "model_type": "diffusion_cond",
    "training": {
        "learning_rate": 2e-4,
        "use_ema": true,
        "cfg_dropout_prob": 0.1,
        "timestep_sampler": "uniform",
        "mask_padding": true,
        "mask_padding_dropout": 0.2,
        "demo": {
            "demo_every": 1000,
            "demo_steps": 100
        }
    },
    "sample_rate": 44100
}
```

### ARC Training
```json
{
    "model_type": "diffusion_cond",
    "training": {
        "arc": {
            "use_model_as_teacher": true,
            "teacher_model_ckpt": "/path/to/teacher.ckpt",
            "use_model_as_discriminator": true
        },
        "use_ema": true,
        "cfg_dropout_prob": 0.15,
        "clip_grad_norm": 1.0
    },
    "sample_rate": 48000
}
```

### Language Model Training
```json
{
    "model_type": "lm",
    "training": {
        "learning_rate": 1e-4,
        "use_ema": false,
        "pre_encoded": true,
        "optimizer_configs": {
            "generator": {
                "type": "AdamW",
                "config": {
                    "lr": 1e-4,
                    "weight_decay": 0.01
                }
            }
        }
    },
    "sample_rate": 44100
}
```

## Dependencies

The training factory depends on:
- `stable_audio_tools.training.autoencoders` - Autoencoder training classes
- `stable_audio_tools.training.diffusion` - Diffusion training classes
- `stable_audio_tools.training.arc` - ARC training classes
- `stable_audio_tools.training.lm` - Language model training classes
- `stable_audio_tools.models.factory` - Model factory for EMA and teacher models

## Migration Notes

### Current Implementation Issues
- Hard-coded model type strings throughout code
- Complex conditional logic for different training types
- Inconsistent parameter validation across model types
- Manual EMA model creation with potential for errors

### Pydantic Migration Benefits
- Type-safe enums for model types and configuration options
- Proper validation of all training parameters
- Validation of parameter relationships (e.g., teacher model requirements)
- Better error messages for configuration issues
- Documentation of all parameters and their purposes
- Support for discriminated unions based on model type 