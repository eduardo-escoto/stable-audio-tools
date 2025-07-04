# defaults.ini Analysis

## Overview

The `defaults.ini` file provides default configuration values for the `train.py` script using the prefigure library. It serves as the base configuration that can be overridden by command-line arguments.

## File Location
- **File**: `defaults.ini`
- **Purpose**: Default configuration values for training
- **Format**: INI configuration file format
- **Used By**: `train.py` via prefigure library

## Configuration Structure

### File Format
- **Sections**: Uses `[DEFAULTS]` section for all configuration
- **Comments**: Lines starting with `#` are comments
- **Values**: Key-value pairs with various data types

### Current Configuration

```ini
[DEFAULTS]

#name of the run
name = stable_audio_tools

# name of the project
project = None

# the batch size
batch_size = 4

# If `true`, attempts to resume training from latest checkpoint.
# In this case, each run must have unique config filename.
recover = false

# Save top K model checkpoints during training.
save_top_k = -1

# number of nodes to use for training
num_nodes = 1

# Multi-GPU strategy for PyTorch Lightning
strategy = "auto"

# Precision to use for training
precision = "16-mixed"

# number of CPU workers for the DataLoader
num_workers = 6

# the random seed
seed = 42

# Batches for gradient accumulation
accum_batches = 1

# Number of steps between checkpoints
checkpoint_every = 10000

# Number of steps between validation runs
val_every = -1

# trainer checkpoint file to restart training from
ckpt_path = ''

# model checkpoint file to start a new training run from
pretrained_ckpt_path = ''

# Checkpoint path for the pretransform model if needed
pretransform_ckpt_path = ''

# configuration model specifying model hyperparameters
model_config = ''

# configuration for datasets
dataset_config = ''

# configuration for validation datasets
val_dataset_config = ''

# directory to save the checkpoints in
save_dir = ''

# gradient_clip_val passed into PyTorch Lightning Trainer
gradient_clip_val = 0.0

# remove the weight norm from the pretransform model
remove_pretransform_weight_norm = ''

# Logger type to use
logger = 'wandb'
```

## Configuration Categories

### Run Identification
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"stable_audio_tools"` | Name of the training run |
| `project` | `str` | `None` | Name of the project (for logging) |

### Training Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `4` | Batch size for training |
| `num_workers` | `int` | `6` | Number of CPU workers for DataLoader |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `accum_batches` | `int` | `1` | Batches for gradient accumulation |
| `gradient_clip_val` | `float` | `0.0` | Gradient clipping value (0.0 = no clipping) |

### Checkpointing & Recovery
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recover` | `bool` | `false` | Attempt to resume training from latest checkpoint |
| `save_top_k` | `int` | `-1` | Save top K model checkpoints (-1 = all) |
| `checkpoint_every` | `int` | `10000` | Number of steps between checkpoints |
| `save_dir` | `str` | `''` | Directory to save checkpoints |
| `ckpt_path` | `str` | `''` | Trainer checkpoint file to restart training from |
| `pretrained_ckpt_path` | `str` | `''` | Model checkpoint file to start new training from |
| `pretransform_ckpt_path` | `str` | `''` | Checkpoint path for pretransform model |

### Configuration Files
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_config` | `str` | `''` | Path to model configuration JSON file |
| `dataset_config` | `str` | `''` | Path to dataset configuration JSON file |
| `val_dataset_config` | `str` | `''` | Path to validation dataset configuration JSON file |

### Distributed Training
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | `int` | `1` | Number of nodes for distributed training |
| `strategy` | `str` | `"auto"` | Multi-GPU strategy for PyTorch Lightning |
| `precision` | `str` | `"16-mixed"` | Precision to use for training |

### Validation
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `val_every` | `int` | `-1` | Number of steps between validation runs (-1 = no validation) |

### Logging
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logger` | `str` | `'wandb'` | Logger type to use ('wandb', 'comet', or None) |

### Special Options
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remove_pretransform_weight_norm` | `str` | `''` | Remove weight norm from pretransform ('pre_load', 'post_load', or '') |

## Data Types and Validation

### String Parameters
- **Empty Strings**: `''` represents unset/optional parameters
- **Quoted Strings**: Some strings are quoted (`"auto"`, `"16-mixed"`)
- **Path Parameters**: Configuration file paths are typically empty by default

### Boolean Parameters
- **Format**: Uses lowercase `true`/`false` (INI standard)
- **Default**: Most boolean flags default to `false`

### Numeric Parameters
- **Integers**: Used for counts, steps, and IDs
- **Floats**: Used for rates and scaling factors
- **Special Values**: `-1` often means "disabled" or "unlimited"

## Usage Patterns

### Default Behavior
- **Empty Paths**: Configuration files must be specified via command line
- **Conservative Defaults**: Settings favor stability over performance
- **Logging Enabled**: WandB logging enabled by default

### Common Overrides
```bash
# Override batch size and save directory
python train.py --batch-size 8 --save-dir ./my_checkpoints

# Change logging
python train.py --logger comet --name my_experiment

# Distributed training
python train.py --num-nodes 4 --strategy deepspeed
```

## Pydantic Schema Design

### INI Configuration Schema

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from enum import Enum

class LoggerType(str, Enum):
    WANDB = "wandb"
    COMET = "comet"
    NONE = "none"

class StrategyType(str, Enum):
    AUTO = "auto"
    DDP = "ddp"
    DDP_FIND_UNUSED = "ddp_find_unused_parameters_true"
    DEEPSPEED = "deepspeed"

class PrecisionType(str, Enum):
    FLOAT32 = "32"
    FLOAT16 = "16"
    BFLOAT16 = "bf16"
    MIXED_16 = "16-mixed"
    MIXED_BF16 = "bf16-mixed"

class WeightNormRemoval(str, Enum):
    NONE = ""
    PRE_LOAD = "pre_load"
    POST_LOAD = "post_load"

class DefaultsConfig(BaseModel):
    """Configuration matching defaults.ini structure"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    # Run identification
    name: str = Field("stable_audio_tools", description="Name of the training run")
    project: Optional[str] = Field(None, description="Name of the project")
    
    # Training parameters
    batch_size: int = Field(4, ge=1, description="Batch size for training")
    num_workers: int = Field(6, ge=0, description="Number of CPU workers for DataLoader")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    accum_batches: int = Field(1, ge=1, description="Batches for gradient accumulation")
    gradient_clip_val: float = Field(0.0, ge=0.0, description="Gradient clipping value")
    
    # Checkpointing & recovery
    recover: bool = Field(False, description="Attempt to resume training from latest checkpoint")
    save_top_k: int = Field(-1, ge=-1, description="Save top K model checkpoints")
    checkpoint_every: int = Field(10000, ge=1, description="Number of steps between checkpoints")
    save_dir: str = Field("", description="Directory to save checkpoints")
    ckpt_path: str = Field("", description="Trainer checkpoint file to restart training from")
    pretrained_ckpt_path: str = Field("", description="Model checkpoint file to start new training from")
    pretransform_ckpt_path: str = Field("", description="Checkpoint path for pretransform model")
    
    # Configuration files
    model_config: str = Field("", description="Path to model configuration JSON file")
    dataset_config: str = Field("", description="Path to dataset configuration JSON file")
    val_dataset_config: str = Field("", description="Path to validation dataset configuration JSON file")
    
    # Distributed training
    num_nodes: int = Field(1, ge=1, description="Number of nodes for distributed training")
    strategy: StrategyType = Field(StrategyType.AUTO, description="Multi-GPU strategy")
    precision: PrecisionType = Field(PrecisionType.MIXED_16, description="Precision for training")
    
    # Validation
    val_every: int = Field(-1, ge=-1, description="Number of steps between validation runs")
    
    # Logging
    logger: LoggerType = Field(LoggerType.WANDB, description="Logger type to use")
    
    # Special options
    remove_pretransform_weight_norm: WeightNormRemoval = Field(
        WeightNormRemoval.NONE, 
        description="Remove weight norm from pretransform"
    )
    
    @validator('val_every')
    def validate_val_every(cls, v):
        if v == 0:
            raise ValueError("val_every cannot be 0 (use -1 for no validation)")
        return v
```

## Hydra Migration Structure

### Base Configuration
```yaml
# config/training/base.yaml
defaults:
  - _self_

# Run identification
name: "stable_audio_tools"
project: null

# Training parameters
batch_size: 4
num_workers: 6
seed: 42
accum_batches: 1
gradient_clip_val: 0.0

# Checkpointing & recovery
recover: false
save_top_k: -1
checkpoint_every: 10000
save_dir: ""
ckpt_path: ""
pretrained_ckpt_path: ""
pretransform_ckpt_path: ""

# Configuration files
model_config: ""
dataset_config: ""
val_dataset_config: ""

# Distributed training
num_nodes: 1
strategy: "auto"
precision: "16-mixed"

# Validation
val_every: -1

# Logging
logger: "wandb"

# Special options
remove_pretransform_weight_norm: ""
```

### Configuration Groups
```yaml
# config/training/strategy/auto.yaml
strategy: "auto"

# config/training/strategy/ddp.yaml
strategy: "ddp"

# config/training/strategy/deepspeed.yaml
strategy: "deepspeed"

# config/training/logger/wandb.yaml
logger: "wandb"

# config/training/logger/comet.yaml
logger: "comet"

# config/training/logger/none.yaml
logger: null
```

## Migration Notes

### From INI to YAML
1. **Section Removal**: Remove `[DEFAULTS]` section header
2. **Boolean Format**: Change `true`/`false` to YAML boolean format
3. **Null Values**: Convert `None` strings to YAML `null`
4. **String Quoting**: Maintain quoted strings where necessary

### Value Conversions
| INI Format | YAML Format | Notes |
|------------|-------------|-------|
| `true` | `true` | Boolean values |
| `false` | `false` | Boolean values |
| `None` | `null` | Null values |
| `''` | `""` | Empty strings |
| `"auto"` | `"auto"` | Quoted strings |

### Validation Improvements
- **Type Safety**: Strong typing with Pydantic
- **Range Validation**: Numeric constraints (e.g., `ge=1`)
- **Enum Validation**: Limited choices for strategy, logger, etc.
- **Cross-field Validation**: Logic validation between related fields

### Backward Compatibility
- **Same Defaults**: Maintain identical default values
- **Same Behavior**: Preserve existing training behavior
- **Migration Tool**: Provide tool to convert existing INI files

## Dependencies

The defaults.ini configuration depends on:
- **Prefigure Library**: For INI file parsing and CLI integration
- **PyTorch Lightning**: For trainer configuration options
- **Logging Libraries**: WandB, Comet for experiment tracking
- **File System**: For checkpoint and configuration file paths 