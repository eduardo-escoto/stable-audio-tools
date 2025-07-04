# pre_encode.py Analysis

## Overview

The `pre_encode.py` script is used to pre-encode audio datasets into latent representations using trained autoencoder models. This preprocessing step can significantly speed up training by avoiding repeated encoding during the training process.

## Script Location
- **File**: `pre_encode.py`
- **Purpose**: Pre-encode audio datasets to latent representations
- **Configuration**: Uses argparse for command-line arguments

## Configuration System

### Current Implementation
- **Argument Parser**: Uses `argparse.ArgumentParser` for CLI parsing
- **JSON Configs**: Loads separate JSON files for model and dataset configurations
- **No INI Files**: Does not use INI configuration files (unlike train.py)

### Configuration Dependencies

#### Required JSON Files
1. **Model Configuration**: `--model-config` argument
   - Contains complete model architecture specification
   - Must be compatible with the checkpoint being loaded
   - Same format as used in training

2. **Dataset Configuration**: `--dataset-config` argument
   - Contains dataset specification and preprocessing parameters
   - Defines which audio files to process
   - Includes preprocessing settings

#### Required Checkpoint
- **Model Checkpoint**: `--ckpt-path` argument
  - Path to trained autoencoder model checkpoint
  - Must be compatible with the model configuration

## Command Line Arguments

### Model Configuration
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--model-config` | `str` | `None` | No* | Path to model configuration JSON file |
| `--ckpt-path` | `str` | `None` | No* | Path to unwrapped autoencoder model checkpoint |

*Note: Either model-config + ckpt-path OR pretrained-name must be provided

### Dataset Configuration
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--dataset-config` | `str` | `None` | Yes | Path to dataset configuration JSON file |
| `--output-path` | `str` | `None` | Yes | Path to output folder for encoded latents |

### Processing Parameters
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--batch-size` | `int` | `1` | No | Batch size for processing |
| `--sample-size` | `int` | `1320960` | No | Number of audio samples to pad/crop to |
| `--num-workers` | `int` | `4` | No | Number of dataloader workers |
| `--shuffle` | `bool` | `False` | No | Whether to shuffle the dataset |

### Model Options
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--model-half` | `bool` | `False` | No | Whether to use half precision (fp16) |
| `--is-discrete` | `bool` | `False` | No | Whether the model uses discrete latents |

### Distributed Processing
| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--num-nodes` | `int` | `1` | No | Number of GPU nodes for distributed processing |
| `--strategy` | `str` | `'auto'` | No | PyTorch Lightning strategy for distributed processing |
| `--limit-batches` | `int` | `None` | No | Limit number of batches to process (for testing) |

## Processing Flow

### 1. Configuration Loading
```python
with open(args.model_config) as f:
    model_config = json.load(f)

with open(args.dataset_config) as f:
    dataset_config = json.load(f)
```

### 2. Model Loading
```python
model, model_config = load_model(
    model_config=model_config,
    model_ckpt_path=args.ckpt_path,
    model_half=args.model_half
)
```

### 3. Dataset Creation
```python
data_loader = create_dataloader_from_config(
    dataset_config,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    sample_rate=model_config["sample_rate"],
    sample_size=args.sample_size,
    audio_channels=model_config.get("audio_channels", 2),
    shuffle=args.shuffle
)
```

### 4. Encoding Process
```python
# For continuous latents
latents = model.encode(audio)

# For discrete latents  
_, info = model.encode(audio, return_info=True)
latents = info[model.bottleneck.tokens_id]
```

### 5. Output Saving
- **Latents**: Saved as `.npy` files
- **Metadata**: Saved as `.json` files
- **Directory Structure**: Organized by GPU rank for distributed processing

## Output Structure

### File Organization
```
output_path/
├── details.json          # Processing details and configuration
├── 0/                    # Rank 0 outputs
│   ├── 000000000.npy    # Latent tensor
│   ├── 000000000.json   # Metadata
│   ├── 000000001.npy
│   └── 000000001.json
├── 1/                    # Rank 1 outputs (if distributed)
│   ├── 001000000.npy
│   └── 001000000.json
└── ...
```

### File Naming Convention
- **Pattern**: `{rank:03d}{batch_idx:06d}{sample_idx:04d}`
- **Example**: `000000001.npy` = rank 0, batch 0, sample 1

### Metadata Format
```json
{
    "audio_channels": 2,
    "sample_rate": 44100,
    "padding_mask": [1, 1, 1, ..., 0, 0],
    "path": "/path/to/original/audio.wav",
    "duration": 30.0,
    "start_time": 0.0,
    "end_time": 30.0
}
```

## Special Features

### Distributed Processing
- **Multi-GPU**: Supports distributed processing across multiple GPUs
- **Multi-Node**: Supports multi-node distributed processing
- **Rank-Based Output**: Each GPU rank saves to separate subdirectory

### Memory Management
- **Garbage Collection**: Explicit garbage collection between batches
- **CUDA Cache**: Clears CUDA cache between batches
- **Half Precision**: Optional fp16 processing to reduce memory usage

### Preprocessing
- **Padding Mask**: Maintains padding information for variable-length audio
- **Interpolation**: Interpolates padding masks to match latent dimensions
- **Tensor Conversion**: Converts all tensors to serializable formats

## Error Handling

### Configuration Validation
- **Required Arguments**: Validates that required arguments are provided
- **File Existence**: Checks that configuration files exist
- **Model Compatibility**: Ensures model and checkpoint are compatible

### Processing Robustness
- **Memory Management**: Handles out-of-memory situations gracefully
- **File I/O**: Robust file writing with error handling
- **Metadata Conversion**: Handles tensor-to-list conversion for JSON serialization

## Pydantic Schema Design

### Pre-encoding Configuration Schema

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from enum import Enum

class PreEncodeConfig(BaseModel):
    """Configuration for pre-encoding script"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    # Model configuration
    model_config_path: Optional[str] = Field(None, description="Path to model configuration JSON file")
    ckpt_path: Optional[str] = Field(None, description="Path to unwrapped autoencoder model checkpoint")
    
    # Dataset configuration
    dataset_config_path: str = Field(..., description="Path to dataset configuration JSON file")
    output_path: str = Field(..., description="Path to output folder for encoded latents")
    
    # Processing parameters
    batch_size: int = Field(1, ge=1, description="Batch size for processing")
    sample_size: int = Field(1320960, ge=1, description="Number of audio samples to pad/crop to")
    num_workers: int = Field(4, ge=0, description="Number of dataloader workers")
    shuffle: bool = Field(False, description="Whether to shuffle the dataset")
    
    # Model options
    model_half: bool = Field(False, description="Whether to use half precision")
    is_discrete: bool = Field(False, description="Whether the model uses discrete latents")
    
    # Distributed processing
    num_nodes: int = Field(1, ge=1, description="Number of GPU nodes")
    strategy: str = Field("auto", description="PyTorch Lightning strategy")
    limit_batches: Optional[int] = Field(None, ge=1, description="Limit number of batches to process")
    
    @validator('model_config_path', 'ckpt_path')
    def validate_model_inputs(cls, v, values):
        # Either both model_config_path and ckpt_path must be provided, or neither
        model_config = values.get('model_config_path')
        ckpt_path = values.get('ckpt_path')
        
        if model_config is None and ckpt_path is None:
            # Both are None - this is OK if using pretrained model
            pass
        elif model_config is None or ckpt_path is None:
            # Only one is None - this is an error
            raise ValueError("Both model_config_path and ckpt_path must be provided together")
        
        return v
    
    @validator('sample_size')
    def validate_sample_size(cls, v):
        # Should be a reasonable audio length
        if v < 1000:  # Less than ~0.02 seconds at 44kHz
            raise ValueError("sample_size is too small")
        if v > 44100 * 300:  # More than 5 minutes at 44kHz
            raise ValueError("sample_size is too large")
        return v
```

### Hydra Migration Structure

```yaml
# config/pre_encode/base.yaml
defaults:
  - _self_
  - model: ???
  - dataset: ???

# Model configuration
model_config_path: null
ckpt_path: null

# Dataset configuration
dataset_config_path: ???
output_path: ???

# Processing parameters
batch_size: 1
sample_size: 1320960
num_workers: 4
shuffle: false

# Model options
model_half: false
is_discrete: false

# Distributed processing
num_nodes: 1
strategy: "auto"
limit_batches: null
```

## Usage Examples

### Basic Pre-encoding
```bash
python pre_encode.py \
    --model-config configs/model_configs/autoencoder.json \
    --ckpt-path checkpoints/autoencoder.ckpt \
    --dataset-config configs/dataset_configs/local_audio.json \
    --output-path ./encoded_latents \
    --batch-size 4
```

### Distributed Pre-encoding
```bash
python pre_encode.py \
    --model-config configs/model_configs/autoencoder.json \
    --ckpt-path checkpoints/autoencoder.ckpt \
    --dataset-config configs/dataset_configs/s3_dataset.json \
    --output-path ./encoded_latents \
    --num-nodes 4 \
    --batch-size 16
```

### Half-precision Processing
```bash
python pre_encode.py \
    --model-config configs/model_configs/autoencoder.json \
    --ckpt-path checkpoints/autoencoder.ckpt \
    --dataset-config configs/dataset_configs/local_audio.json \
    --output-path ./encoded_latents \
    --model-half \
    --batch-size 8
```

### Discrete Model Processing
```bash
python pre_encode.py \
    --model-config configs/model_configs/discrete_autoencoder.json \
    --ckpt-path checkpoints/discrete_autoencoder.ckpt \
    --dataset-config configs/dataset_configs/local_audio.json \
    --output-path ./encoded_tokens \
    --is-discrete \
    --batch-size 2
```

## Migration Notes

### From Argparse to Hydra
1. **Argument Structure**: Maintain same argument names for compatibility
2. **Configuration Composition**: Use Hydra's composition for different model/dataset combinations
3. **Validation**: Add comprehensive validation with Pydantic
4. **Default Values**: Maintain same default values as current implementation

### Breaking Changes
- Configuration file format changes from command-line only to YAML-based
- Some argument validation will be stricter
- Output format remains the same

### Backward Compatibility
- Maintain same CLI argument names
- Same output file format and structure
- Same processing behavior and results

## Dependencies

The pre_encode.py script depends on:
- `stable_audio_tools.models.factory` - For model creation
- `stable_audio_tools.models.utils` - For checkpoint loading
- `stable_audio_tools.data.dataset` - For dataset creation
- `stable_audio_tools.models.pretrained` - For pretrained model loading
- PyTorch Lightning - For distributed processing framework 