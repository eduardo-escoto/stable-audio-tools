# train.py Analysis

## Overview

The `train.py` script is the main training entry point for stable-audio-tools. It uses the prefigure library to parse command-line arguments and INI files, loads model and dataset configurations, and orchestrates the training process using PyTorch Lightning.

## Script Location
- **File**: `train.py`
- **Purpose**: Main training script for all model types
- **Configuration**: Uses `defaults.ini` and prefigure library

## Configuration System

### Current Implementation
- **Argument Parser**: Uses `prefigure.get_all_args()` to parse arguments
- **INI Configuration**: Loads defaults from `defaults.ini`
- **JSON Configs**: Loads separate JSON files for model and dataset configurations
- **Command Line Overrides**: CLI arguments override INI file values

### Configuration Dependencies

#### Required JSON Files
1. **Model Configuration**: `--model-config` argument
   - Contains complete model architecture specification
   - Includes model type, architecture parameters, and training settings
   - Structure depends on model type (autoencoder, diffusion, etc.)

2. **Dataset Configuration**: `--dataset-config` argument
   - Contains dataset specification and preprocessing parameters
   - Includes dataset type, paths, and data loading parameters

3. **Validation Dataset Configuration**: `--val-dataset-config` argument (optional)
   - Similar to dataset config but for validation data
   - Used for validation and demo generation

## Command Line Arguments

### Configuration Files
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--model-config` | `str` | `''` | Yes | Path to model configuration JSON file |
| `--dataset-config` | `str` | `''` | Yes | Path to dataset configuration JSON file |
| `--val-dataset-config` | `str` | `''` | No | Path to validation dataset configuration JSON file |

### Training Parameters
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--batch-size` | `int` | `4` | No | Batch size for training |
| `--num-workers` | `int` | `6` | No | Number of CPU workers for DataLoader |
| `--seed` | `int` | `42` | No | Random seed for reproducibility |
| `--accum-batches` | `int` | `1` | No | Batches for gradient accumulation |
| `--gradient-clip-val` | `float` | `0.0` | No | Gradient clipping value (0.0 = no clipping) |

### Checkpointing
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--checkpoint-every` | `int` | `10000` | No | Number of steps between checkpoints |
| `--save-dir` | `str` | `''` | No | Directory to save checkpoints |
| `--save-top-k` | `int` | `-1` | No | Save top K model checkpoints (-1 = all) |
| `--ckpt-path` | `str` | `''` | No | Trainer checkpoint file to restart training from |
| `--pretrained-ckpt-path` | `str` | `''` | No | Model checkpoint file to start new training run from |
| `--pretransform-ckpt-path` | `str` | `''` | No | Checkpoint path for pretransform model |

### Distributed Training
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--num-nodes` | `int` | `1` | No | Number of nodes for distributed training |
| `--strategy` | `str` | `"auto"` | No | Multi-GPU strategy for PyTorch Lightning |
| `--precision` | `str` | `"16-mixed"` | No | Precision for training |

### Validation
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--val-every` | `int` | `-1` | No | Number of steps between validation runs (-1 = no validation) |

### Logging
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--logger` | `str` | `'wandb'` | No | Logger type ('wandb', 'comet', or None) |
| `--name` | `str` | `'stable_audio_tools'` | No | Name of the run/project |
| `--project` | `str` | `None` | No | Name of the project |

### Special Options
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--recover` | `bool` | `False` | No | Attempt to resume training from latest checkpoint |
| `--remove-pretransform-weight-norm` | `str` | `''` | No | Remove weight norm from pretransform ('pre_load', 'post_load', or '') |

## Configuration Flow

### 1. Argument Loading
```python
args = get_all_args()  # Loads from defaults.ini + CLI overrides
```

### 2. Model Configuration Loading
```python
with open(args.model_config) as f:
    model_config = json.load(f)
```

### 3. Dataset Configuration Loading
```python
with open(args.dataset_config) as f:
    dataset_config = json.load(f)
```

### 4. Model Creation
```python
model = create_model_from_config(model_config)
```

### 5. DataLoader Creation
```python
train_dl = create_dataloader_from_config(
    dataset_config,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    sample_rate=model_config["sample_rate"],
    sample_size=model_config["sample_size"],
    audio_channels=model_config.get("audio_channels", 2),
)
```

### 6. Training Wrapper Creation
```python
training_wrapper = create_training_wrapper_from_config(model_config, model)
```

## Special Features

### SLURM Integration
- **Seed Adjustment**: If `SLURM_PROCID` environment variable is set, adds process ID to seed
- **Multi-node Support**: Uses `--num-nodes` for distributed training

### Logger Integration
- **WandB**: Automatic project/run organization, checkpoint directory creation
- **Comet**: Alternative logging with similar features
- **Configuration Logging**: Pushes complete configuration to logging service

### Checkpoint Management
- **Automatic Saving**: Saves checkpoints every `checkpoint_every` steps
- **Model Config Embedding**: Embeds model configuration in checkpoint files
- **Directory Organization**: Organizes checkpoints by logger project/run ID

### Demo Generation
- **Automatic Demos**: Generates audio demos during training
- **Validation Data**: Uses validation dataset for demo generation if available
- **Fallback**: Uses training data for demos if no validation data provided

## Error Handling

### Configuration Validation
- **Required Files**: Asserts that model and dataset config files exist
- **Model Requirements**: Validates required fields in model config
- **Dataset Compatibility**: Ensures dataset config matches model requirements

### Training Robustness
- **Exception Callback**: Catches and logs training exceptions
- **Gradient Clipping**: Optional gradient clipping for stability
- **Checkpoint Recovery**: Supports resuming from interrupted training

## Environment Variables

### SLURM Support
- `SLURM_PROCID`: Process ID for multi-node training (adds to seed)

### PyTorch Lightning
- Standard PyTorch Lightning environment variables are respected

## Pydantic Schema Design

### Training Configuration Schema

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

class TrainingConfig(BaseModel):
    """Configuration for training script"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    # Configuration files
    model_config_path: str = Field(..., description="Path to model configuration JSON file")
    dataset_config_path: str = Field(..., description="Path to dataset configuration JSON file")
    val_dataset_config_path: Optional[str] = Field(None, description="Path to validation dataset configuration JSON file")
    
    # Training parameters
    batch_size: int = Field(4, ge=1, description="Batch size for training")
    num_workers: int = Field(6, ge=0, description="Number of CPU workers for DataLoader")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    accum_batches: int = Field(1, ge=1, description="Batches for gradient accumulation")
    gradient_clip_val: float = Field(0.0, ge=0.0, description="Gradient clipping value")
    
    # Checkpointing
    checkpoint_every: int = Field(10000, ge=1, description="Number of steps between checkpoints")
    save_dir: Optional[str] = Field(None, description="Directory to save checkpoints")
    save_top_k: int = Field(-1, ge=-1, description="Save top K model checkpoints")
    ckpt_path: Optional[str] = Field(None, description="Trainer checkpoint file to restart training from")
    pretrained_ckpt_path: Optional[str] = Field(None, description="Model checkpoint file to start new training run from")
    pretransform_ckpt_path: Optional[str] = Field(None, description="Checkpoint path for pretransform model")
    
    # Distributed training
    num_nodes: int = Field(1, ge=1, description="Number of nodes for distributed training")
    strategy: StrategyType = Field(StrategyType.AUTO, description="Multi-GPU strategy")
    precision: PrecisionType = Field(PrecisionType.MIXED_16, description="Training precision")
    
    # Validation
    val_every: int = Field(-1, ge=-1, description="Number of steps between validation runs")
    
    # Logging
    logger: LoggerType = Field(LoggerType.WANDB, description="Logger type")
    name: str = Field("stable_audio_tools", description="Name of the run")
    project: Optional[str] = Field(None, description="Name of the project")
    
    # Special options
    recover: bool = Field(False, description="Attempt to resume training from latest checkpoint")
    remove_pretransform_weight_norm: WeightNormRemoval = Field(WeightNormRemoval.NONE, description="Remove weight norm from pretransform")
    
    @validator('save_dir')
    def validate_save_dir(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("save_dir must be a string path")
        return v
    
    @validator('val_every')
    def validate_val_every(cls, v):
        if v == 0:
            raise ValueError("val_every cannot be 0 (use -1 for no validation)")
        return v
```

### Hydra Migration Structure

```yaml
# config/training/base.yaml
defaults:
  - _self_
  - model: ???
  - dataset: ???
  - logger: wandb
  - strategy: auto

# Training parameters
batch_size: 4
num_workers: 6
seed: 42
accum_batches: 1
gradient_clip_val: 0.0

# Checkpointing
checkpoint_every: 10000
save_dir: null
save_top_k: -1
ckpt_path: null
pretrained_ckpt_path: null
pretransform_ckpt_path: null

# Distributed training
num_nodes: 1
precision: "16-mixed"

# Validation
val_every: -1

# Logging
name: "stable_audio_tools"
project: null

# Special options
recover: false
remove_pretransform_weight_norm: ""
```

## Usage Examples

### Basic Training
```bash
python train.py \
    --model-config configs/model_configs/autoencoder.json \
    --dataset-config configs/dataset_configs/local_audio.json \
    --batch-size 8 \
    --save-dir ./checkpoints
```

### Distributed Training
```bash
python train.py \
    --model-config configs/model_configs/diffusion_cond.json \
    --dataset-config configs/dataset_configs/s3_dataset.json \
    --num-nodes 4 \
    --strategy deepspeed \
    --batch-size 32 \
    --accum-batches 4
```

### Resume Training
```bash
python train.py \
    --model-config configs/model_configs/autoencoder.json \
    --dataset-config configs/dataset_configs/local_audio.json \
    --ckpt-path ./checkpoints/last.ckpt \
    --recover
```

## Migration Notes

### From Prefigure to Hydra
1. **INI to YAML**: Convert `defaults.ini` to Hydra YAML configuration
2. **Argument Structure**: Maintain same argument names for compatibility
3. **Configuration Composition**: Use Hydra's composition features for different model/dataset combinations
4. **Validation**: Add comprehensive validation with Pydantic
5. **Documentation**: Self-documenting configuration structure

### Breaking Changes
- Configuration file format changes from INI to YAML
- Some argument names may change for consistency
- Validation will catch previously accepted invalid configurations

### Backward Compatibility
- Maintain same CLI argument names where possible
- Provide migration tools to convert existing configurations
- Support both old and new configuration formats during transition 