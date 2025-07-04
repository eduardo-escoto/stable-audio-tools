# Training Utils Factory Analysis

## Overview

The training utils factory in `stable_audio_tools/training/utils.py` creates optimizers and learning rate schedulers from configuration. These utilities provide flexible optimizer and scheduler setup for training different models.

## Factory Functions

### `create_optimizer_from_config(optimizer_config, parameters)`

**Purpose**: Creates PyTorch optimizers from configuration
**Location**: `stable_audio_tools/training/utils.py:56`

### `create_scheduler_from_config(scheduler_config, optimizer)`

**Purpose**: Creates learning rate schedulers from configuration
**Location**: `stable_audio_tools/training/utils.py:77`

## Function Parameters

### Optimizer Factory Parameters

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `optimizer_config` | `dict` | Optimizer configuration | Must contain type and config |
| `parameters` | `iterable` | Model parameters to optimize | Must be valid parameter iterable |

### Scheduler Factory Parameters

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `scheduler_config` | `dict` | Scheduler configuration | Must contain type and config |
| `optimizer` | `torch.optim.Optimizer` | Optimizer instance | Must be valid optimizer |

## Configuration Schema

### Optimizer Configuration
```python
{
    "type": str,  # Required - optimizer type
    "config": dict,  # Required - optimizer-specific configuration
}
```

### Scheduler Configuration
```python
{
    "type": str,  # Required - scheduler type
    "config": dict,  # Required - scheduler-specific configuration
}
```

## Supported Optimizers

### 1. `FusedAdam` - DeepSpeed Fused Adam
- **Class**: `deepspeed.ops.adam.FusedAdam`
- **Purpose**: High-performance fused Adam optimizer
- **Module**: `deepspeed.ops.adam`

#### Configuration
```python
{
    "type": "FusedAdam",
    "config": {
        "lr": float,  # Required - learning rate
        "betas": tuple,  # Optional - beta coefficients
        "eps": float,  # Optional - epsilon value
        "weight_decay": float,  # Optional - weight decay
        "amsgrad": bool,  # Optional - use AMSGrad variant
        "adam_w_mode": bool,  # Optional - use AdamW mode
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `lr` | `float` | **Required** | Learning rate | Must be positive |
| `betas` | `tuple` | `(0.9, 0.999)` | Beta coefficients | Must be valid beta values |
| `eps` | `float` | `1e-8` | Epsilon value | Must be positive |
| `weight_decay` | `float` | `0` | Weight decay | Must be non-negative |
| `amsgrad` | `bool` | `False` | Use AMSGrad variant | Boolean |
| `adam_w_mode` | `bool` | `True` | Use AdamW mode | Boolean |

### 2. Standard PyTorch Optimizers
- **Source**: `torch.optim` module
- **Purpose**: Standard PyTorch optimizers
- **Supported Types**: Any optimizer from `torch.optim`

#### Common Optimizer Types

##### `Adam` - Adam Optimizer
```python
{
    "type": "Adam",
    "config": {
        "lr": float,  # Required - learning rate
        "betas": tuple,  # Optional - beta coefficients
        "eps": float,  # Optional - epsilon value
        "weight_decay": float,  # Optional - weight decay
        "amsgrad": bool,  # Optional - use AMSGrad variant
    }
}
```

##### `AdamW` - AdamW Optimizer
```python
{
    "type": "AdamW",
    "config": {
        "lr": float,  # Required - learning rate
        "betas": tuple,  # Optional - beta coefficients
        "eps": float,  # Optional - epsilon value
        "weight_decay": float,  # Optional - weight decay
        "amsgrad": bool,  # Optional - use AMSGrad variant
    }
}
```

##### `SGD` - Stochastic Gradient Descent
```python
{
    "type": "SGD",
    "config": {
        "lr": float,  # Required - learning rate
        "momentum": float,  # Optional - momentum factor
        "dampening": float,  # Optional - dampening for momentum
        "weight_decay": float,  # Optional - weight decay
        "nesterov": bool,  # Optional - use Nesterov momentum
    }
}
```

## Supported Schedulers

### 1. `InverseLR` - Custom Inverse Learning Rate Scheduler
- **Class**: `InverseLR` (custom implementation)
- **Purpose**: Inverse decay learning rate schedule with optional exponential warmup
- **Module**: `stable_audio_tools.training.utils`

#### Configuration
```python
{
    "type": "InverseLR",
    "config": {
        "inv_gamma": float,  # Optional - inverse multiplicative factor
        "power": float,  # Optional - exponential factor
        "warmup": float,  # Optional - exponential warmup factor
        "final_lr": float,  # Optional - final learning rate
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `inv_gamma` | `float` | `1.0` | Inverse multiplicative factor | Must be positive |
| `power` | `float` | `1.0` | Exponential factor | Must be positive |
| `warmup` | `float` | `0.0` | Exponential warmup factor | Must be 0 ≤ warmup < 1 |
| `final_lr` | `float` | `0.0` | Final learning rate | Must be non-negative |

#### Formula
The InverseLR scheduler implements:
```python
warmup = 1 - warmup ** (step + 1)
lr_mult = (1 + step / inv_gamma) ** -power
lr = warmup * max(final_lr, base_lr * lr_mult)
```

### 2. Standard PyTorch Schedulers
- **Source**: `torch.optim.lr_scheduler` module
- **Purpose**: Standard PyTorch learning rate schedulers
- **Supported Types**: Any scheduler from `torch.optim.lr_scheduler`

#### Common Scheduler Types

##### `StepLR` - Step Learning Rate Scheduler
```python
{
    "type": "StepLR",
    "config": {
        "step_size": int,  # Required - period of learning rate decay
        "gamma": float,  # Optional - multiplicative factor of decay
    }
}
```

##### `ExponentialLR` - Exponential Learning Rate Scheduler
```python
{
    "type": "ExponentialLR",
    "config": {
        "gamma": float,  # Required - multiplicative factor of decay
    }
}
```

##### `CosineAnnealingLR` - Cosine Annealing Learning Rate Scheduler
```python
{
    "type": "CosineAnnealingLR",
    "config": {
        "T_max": int,  # Required - maximum number of iterations
        "eta_min": float,  # Optional - minimum learning rate
    }
}
```

##### `ReduceLROnPlateau` - Reduce Learning Rate on Plateau
```python
{
    "type": "ReduceLROnPlateau",
    "config": {
        "mode": str,  # Optional - 'min' or 'max'
        "factor": float,  # Optional - factor by which the learning rate will be reduced
        "patience": int,  # Optional - number of epochs with no improvement
        "threshold": float,  # Optional - threshold for measuring the new optimum
        "threshold_mode": str,  # Optional - 'rel' or 'abs'
        "cooldown": int,  # Optional - number of epochs to wait before resuming
        "min_lr": float,  # Optional - minimum learning rate
        "eps": float,  # Optional - minimal decay applied to lr
    }
}
```

## Error Handling

### Optimizer Creation Errors
- **Missing type**: KeyError if 'type' not in config
- **Unknown optimizer type**: AttributeError if optimizer type not found in torch.optim
- **Invalid parameters**: TypeError or ValueError from optimizer constructor

### Scheduler Creation Errors
- **Missing type**: KeyError if 'type' not in config
- **Unknown scheduler type**: AttributeError if scheduler type not found
- **Invalid parameters**: TypeError or ValueError from scheduler constructor
- **Invalid warmup range**: ValueError if warmup not in [0, 1) for InverseLR

## Return Types

### Optimizers
- **Standard PyTorch**: `torch.optim.Optimizer` subclasses
- **FusedAdam**: `deepspeed.ops.adam.FusedAdam`

### Schedulers
- **Standard PyTorch**: `torch.optim.lr_scheduler._LRScheduler` subclasses
- **InverseLR**: `InverseLR` (custom class)

## Pydantic Schema Design

### Base Configuration

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Union, Tuple
from enum import Enum

class OptimizerType(str, Enum):
    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SGD"
    FUSED_ADAM = "FusedAdam"
    RMSPROP = "RMSprop"
    ADAGRAD = "Adagrad"

class SchedulerType(str, Enum):
    INVERSE_LR = "InverseLR"
    STEP_LR = "StepLR"
    EXPONENTIAL_LR = "ExponentialLR"
    COSINE_ANNEALING_LR = "CosineAnnealingLR"
    REDUCE_LR_ON_PLATEAU = "ReduceLROnPlateau"
    MULTI_STEP_LR = "MultiStepLR"
    LAMBDA_LR = "LambdaLR"

class BaseOptimizerConfig(BaseModel):
    """Base configuration for optimizers"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: OptimizerType = Field(..., description="Optimizer type")
    config: Dict[str, Any] = Field(..., description="Optimizer-specific configuration")

class BaseSchedulerConfig(BaseModel):
    """Base configuration for schedulers"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: SchedulerType = Field(..., description="Scheduler type")
    config: Dict[str, Any] = Field(..., description="Scheduler-specific configuration")
```

### Specific Optimizer Configs

```python
class AdamConfig(BaseModel):
    """Configuration for Adam optimizer"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    lr: float = Field(..., description="Learning rate", gt=0)
    betas: Tuple[float, float] = Field((0.9, 0.999), description="Beta coefficients")
    eps: float = Field(1e-8, description="Epsilon value", gt=0)
    weight_decay: float = Field(0.0, description="Weight decay", ge=0)
    amsgrad: bool = Field(False, description="Use AMSGrad variant")
    
    @validator('betas')
    def validate_betas(cls, v):
        if not (0 <= v[0] < 1 and 0 <= v[1] < 1):
            raise ValueError("Beta values must be in [0, 1)")
        return v

class AdamOptimizerConfig(BaseOptimizerConfig):
    type: Literal[OptimizerType.ADAM] = OptimizerType.ADAM
    config: AdamConfig = Field(..., description="Adam configuration")

class FusedAdamConfig(AdamConfig):
    """Configuration for FusedAdam optimizer"""
    adam_w_mode: bool = Field(True, description="Use AdamW mode")

class FusedAdamOptimizerConfig(BaseOptimizerConfig):
    type: Literal[OptimizerType.FUSED_ADAM] = OptimizerType.FUSED_ADAM
    config: FusedAdamConfig = Field(..., description="FusedAdam configuration")

class SGDConfig(BaseModel):
    """Configuration for SGD optimizer"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    lr: float = Field(..., description="Learning rate", gt=0)
    momentum: float = Field(0.0, description="Momentum factor", ge=0)
    dampening: float = Field(0.0, description="Dampening for momentum", ge=0)
    weight_decay: float = Field(0.0, description="Weight decay", ge=0)
    nesterov: bool = Field(False, description="Use Nesterov momentum")
    
    @validator('nesterov')
    def validate_nesterov(cls, v, values):
        if v and values.get('momentum', 0) <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        return v

class SGDOptimizerConfig(BaseOptimizerConfig):
    type: Literal[OptimizerType.SGD] = OptimizerType.SGD
    config: SGDConfig = Field(..., description="SGD configuration")
```

### Specific Scheduler Configs

```python
class InverseLRConfig(BaseModel):
    """Configuration for InverseLR scheduler"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    inv_gamma: float = Field(1.0, description="Inverse multiplicative factor", gt=0)
    power: float = Field(1.0, description="Exponential factor", gt=0)
    warmup: float = Field(0.0, description="Exponential warmup factor", ge=0, lt=1)
    final_lr: float = Field(0.0, description="Final learning rate", ge=0)

class InverseLRSchedulerConfig(BaseSchedulerConfig):
    type: Literal[SchedulerType.INVERSE_LR] = SchedulerType.INVERSE_LR
    config: InverseLRConfig = Field(..., description="InverseLR configuration")

class StepLRConfig(BaseModel):
    """Configuration for StepLR scheduler"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    step_size: int = Field(..., description="Period of learning rate decay", gt=0)
    gamma: float = Field(0.1, description="Multiplicative factor of decay", gt=0, le=1)

class StepLRSchedulerConfig(BaseSchedulerConfig):
    type: Literal[SchedulerType.STEP_LR] = SchedulerType.STEP_LR
    config: StepLRConfig = Field(..., description="StepLR configuration")

class CosineAnnealingLRConfig(BaseModel):
    """Configuration for CosineAnnealingLR scheduler"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    T_max: int = Field(..., description="Maximum number of iterations", gt=0)
    eta_min: float = Field(0.0, description="Minimum learning rate", ge=0)

class CosineAnnealingLRSchedulerConfig(BaseSchedulerConfig):
    type: Literal[SchedulerType.COSINE_ANNEALING_LR] = SchedulerType.COSINE_ANNEALING_LR
    config: CosineAnnealingLRConfig = Field(..., description="CosineAnnealingLR configuration")
```

## Usage Examples

### Adam Optimizer
```json
{
    "type": "Adam",
    "config": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
    }
}
```

### FusedAdam Optimizer
```json
{
    "type": "FusedAdam",
    "config": {
        "lr": 2e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "adam_w_mode": true
    }
}
```

### SGD with Nesterov Momentum
```json
{
    "type": "SGD",
    "config": {
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "nesterov": true
    }
}
```

### InverseLR Scheduler
```json
{
    "type": "InverseLR",
    "config": {
        "inv_gamma": 10000.0,
        "power": 1.0,
        "warmup": 0.1,
        "final_lr": 1e-6
    }
}
```

### CosineAnnealingLR Scheduler
```json
{
    "type": "CosineAnnealingLR",
    "config": {
        "T_max": 100000,
        "eta_min": 1e-6
    }
}
```

### Complete Optimizer and Scheduler Configuration
```json
{
    "optimizer_configs": {
        "generator": {
            "type": "AdamW",
            "config": {
                "lr": 1e-4,
                "weight_decay": 0.01
            }
        },
        "discriminator": {
            "type": "Adam",
            "config": {
                "lr": 2e-4,
                "betas": [0.5, 0.999]
            }
        }
    },
    "scheduler_configs": {
        "generator": {
            "type": "InverseLR",
            "config": {
                "inv_gamma": 10000.0,
                "warmup": 0.1
            }
        },
        "discriminator": {
            "type": "StepLR",
            "config": {
                "step_size": 50000,
                "gamma": 0.5
            }
        }
    }
}
```

## Dependencies

The training utils factory depends on:
- `torch.optim` - PyTorch optimizers
- `torch.optim.lr_scheduler` - PyTorch learning rate schedulers
- `deepspeed.ops.adam` - DeepSpeed FusedAdam optimizer (optional)

## Migration Notes

### Current Implementation Issues
- Dynamic attribute access using `getattr` with potential for AttributeError
- Limited validation of optimizer and scheduler parameters
- No type safety for configuration parameters
- Hard-coded special case for FusedAdam

### Pydantic Migration Benefits
- Type-safe enums for optimizer and scheduler types
- Proper validation of all configuration parameters
- Validation of parameter relationships (e.g., Nesterov momentum requirements)
- Better error messages for configuration issues
- Documentation of all parameters and their purposes
- Support for discriminated unions based on optimizer/scheduler type
- Elimination of runtime AttributeError through compile-time validation 