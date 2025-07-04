# Conditioning Factory Analysis

## Overview

The conditioning factory in `stable_audio_tools/models/conditioners.py` creates multi-modal conditioning systems from configuration. This factory builds complete conditioning pipelines that can handle text, audio, and numeric conditioning for generative models.

## Factory Function

### `create_multi_conditioner_from_conditioning_config(config, pretransform=None)`

**Purpose**: Creates multi-modal conditioners from configuration
**Location**: `stable_audio_tools/models/conditioners.py:685`

## Configuration Schema

### Top-Level Structure
```python
{
    "cond_dim": int,  # Required - conditioning dimension
    "configs": list,  # Required - list of conditioner configurations
    "default_keys": dict,  # Optional - default key mappings
    "pre_encoded_keys": list,  # Optional - pre-encoded conditioning keys
}
```

### Top-Level Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `cond_dim` | `int` | Conditioning dimension | **Required** | Must be positive integer |
| `configs` | `list` | List of conditioner configurations | **Required** | Must contain valid conditioner configs |
| `default_keys` | `dict` | Default key mappings | `{}` | Dictionary of string mappings |
| `pre_encoded_keys` | `list` | Pre-encoded conditioning keys | `[]` | List of strings |

### Individual Conditioner Configuration
```python
{
    "id": str,  # Required - conditioner identifier
    "type": str,  # Required - conditioner type
    "config": dict,  # Required - conditioner-specific configuration
}
```

### Conditioner Configuration Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `id` | `str` | Conditioner identifier | **Required** | Must be unique string |
| `type` | `str` | Conditioner type | **Required** | Must be supported type |
| `config` | `dict` | Type-specific configuration | **Required** | Structure depends on type |

## Supported Conditioner Types

### 1. `t5` - T5 Text Encoder
- **Class**: `T5Conditioner`
- **Purpose**: Text conditioning using T5 language model
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "t5",
    "config": {
        "t5_model_name": str,  # Optional - T5 model name
        "max_length": int,  # Optional - maximum text length
        "enable_grad": bool,  # Optional - whether to enable gradients
        "project_out": bool,  # Optional - whether to project output
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `t5_model_name` | `str` | `"t5-base"` | T5 model name | Must be valid T5 model |
| `max_length` | `int` | `128` | Maximum text length | Must be positive |
| `enable_grad` | `bool` | `False` | Enable T5 gradients | Boolean |
| `project_out` | `bool` | `False` | Project output dimension | Boolean |

### 2. `clap_text` - CLAP Text Encoder
- **Class**: `CLAPTextConditioner`
- **Purpose**: Text conditioning using CLAP model
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "clap_text",
    "config": {
        "clap_ckpt_path": str,  # Required - CLAP checkpoint path
        "use_text_features": bool,  # Optional - use text features
        "feature_layer_ix": int,  # Optional - feature layer index
        "audio_model_type": str,  # Optional - audio model type
        "enable_fusion": bool,  # Optional - enable fusion
        "project_out": bool,  # Optional - project output
        "finetune": bool,  # Optional - enable finetuning
    }
}
```

### 3. `clap_audio` - CLAP Audio Encoder
- **Class**: `CLAPAudioConditioner`
- **Purpose**: Audio conditioning using CLAP model
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "clap_audio",
    "config": {
        "clap_ckpt_path": str,  # Required - CLAP checkpoint path
        "audio_model_type": str,  # Optional - audio model type
        "enable_fusion": bool,  # Optional - enable fusion
        "project_out": bool,  # Optional - project output
    }
}
```

### 4. `int` - Integer Conditioning
- **Class**: `IntConditioner`
- **Purpose**: Integer value conditioning (e.g., class labels)
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "int",
    "config": {
        "min_val": int,  # Optional - minimum value
        "max_val": int,  # Optional - maximum value
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `min_val` | `int` | `0` | Minimum value | Must be ≤ max_val |
| `max_val` | `int` | `512` | Maximum value | Must be ≥ min_val |

### 5. `number` - Number Conditioning
- **Class**: `NumberConditioner`
- **Purpose**: Floating-point number conditioning (e.g., tempo, duration)
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "number",
    "config": {
        "min_val": float,  # Optional - minimum value
        "max_val": float,  # Optional - maximum value
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `min_val` | `float` | `0.0` | Minimum value | Must be ≤ max_val |
| `max_val` | `float` | `1.0` | Maximum value | Must be ≥ min_val |

### 6. `list` - List Conditioning
- **Class**: `ListConditioner`
- **Purpose**: Categorical conditioning from predefined options
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "list",
    "config": {
        "options": list,  # Required - list of valid options
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `options` | `list` | **Required** | Valid option strings | Must be non-empty list |

### 7. `phoneme` - Phoneme Conditioning
- **Class**: `PhonemeConditioner`
- **Purpose**: Phoneme-based text conditioning
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "phoneme",
    "config": {
        "max_length": int,  # Optional - maximum phoneme length
        "project_out": bool,  # Optional - project output
    }
}
```

### 8. `lut` - Tokenizer LUT Conditioning
- **Class**: `TokenizerLUTConditioner`
- **Purpose**: Tokenizer-based text conditioning with lookup table
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "lut",
    "config": {
        "tokenizer_name": str,  # Required - tokenizer name
        "max_length": int,  # Optional - maximum token length
        "use_abs_pos_emb": bool,  # Optional - use absolute positional embedding
        "project_out": bool,  # Optional - project output
        "special_tokens": list,  # Optional - special tokens
    }
}
```

### 9. `pretransform` - Pretransform Conditioning
- **Class**: `PretransformConditioner`
- **Purpose**: Audio conditioning using pretransform models
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "pretransform",
    "config": {
        "sample_rate": int,  # Required - audio sample rate
        "use_model_pretransform": bool,  # Optional - use model's pretransform
        "pretransform_config": dict,  # Required if not using model pretransform
        "pretransform_ckpt_path": str,  # Optional - pretransform checkpoint
        "save_pretransform": bool,  # Optional - save pretransform in module
    }
}
```

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `sample_rate` | `int` | **Required** | Audio sample rate | Must be positive |
| `use_model_pretransform` | `bool` | `False` | Use main model's pretransform | Boolean |
| `pretransform_config` | `dict` | **Required** | Pretransform configuration | Required if not using model pretransform |
| `pretransform_ckpt_path` | `str` | `None` | Pretransform checkpoint path | Optional file path |
| `save_pretransform` | `bool` | `False` | Save pretransform in module | Boolean |

### 10. `source_mix` - Source Mix Conditioning
- **Class**: `SourceMixConditioner`
- **Purpose**: Multi-source audio mixing conditioning
- **Module**: `stable_audio_tools.models.conditioners`

#### Configuration
```python
{
    "type": "source_mix",
    "config": {
        "sample_rate": int,  # Required - audio sample rate
        "use_model_pretransform": bool,  # Optional - use model's pretransform
        "pretransform_config": dict,  # Required if not using model pretransform
        "pretransform_ckpt_path": str,  # Optional - pretransform checkpoint
        "save_pretransform": bool,  # Optional - save pretransform in module
        "source_keys": list,  # Optional - source key names
        "pre_encoded": bool,  # Optional - whether sources are pre-encoded
        "allow_null_source": bool,  # Optional - allow null sources
        "source_length": int,  # Required if allow_null_source is True
    }
}
```

## Error Handling

### Configuration Validation Errors
- **Missing sample_rate (pretransform/source_mix)**: `AssertionError` - "Sample rate must be specified for {type} conditioners"
- **Missing model pretransform**: `AssertionError` - "Model pretransform must be specified for {type} conditioners"
- **Unknown conditioner type**: `ValueError` - "Unknown conditioner type: {conditioner_type}"

### Special Processing Requirements
- **Pretransform Conditioners**: Require either model pretransform or separate pretransform config
- **Source Mix Conditioners**: Require source length if null sources are allowed
- **Output Dimension**: Automatically added to all conditioner configs as `output_dim`

## Return Type

### Multi-Conditioner
- **Class**: `MultiConditioner`
- **Module**: `stable_audio_tools.models.conditioners`
- **Purpose**: Container for multiple conditioners with unified interface

## Pydantic Schema Design

### Base Configuration

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
from enum import Enum

class ConditionerType(str, Enum):
    T5 = "t5"
    CLAP_TEXT = "clap_text"
    CLAP_AUDIO = "clap_audio"
    INT = "int"
    NUMBER = "number"
    LIST = "list"
    PHONEME = "phoneme"
    LUT = "lut"
    PRETRANSFORM = "pretransform"
    SOURCE_MIX = "source_mix"

class BaseConditionerConfig(BaseModel):
    """Base configuration for conditioners"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    id: str = Field(..., description="Conditioner identifier")
    type: ConditionerType = Field(..., description="Conditioner type")
    config: Dict[str, Any] = Field(..., description="Type-specific configuration")

class MultiConditionerConfig(BaseModel):
    """Configuration for multi-modal conditioner"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    cond_dim: int = Field(..., description="Conditioning dimension", gt=0)
    configs: List[BaseConditionerConfig] = Field(..., description="List of conditioner configurations")
    default_keys: Dict[str, str] = Field(default_factory=dict, description="Default key mappings")
    pre_encoded_keys: List[str] = Field(default_factory=list, description="Pre-encoded conditioning keys")
    
    @validator('configs')
    def validate_unique_ids(cls, v):
        ids = [config.id for config in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Conditioner IDs must be unique")
        return v
```

### Specific Conditioner Configs

```python
class T5ConditionerConfig(BaseConditionerConfig):
    type: Literal[ConditionerType.T5] = ConditionerType.T5
    config: Dict[str, Any] = Field(default_factory=lambda: {
        "t5_model_name": "t5-base",
        "max_length": 128,
        "enable_grad": False,
        "project_out": False,
    })

class NumberConditionerConfig(BaseConditionerConfig):
    type: Literal[ConditionerType.NUMBER] = ConditionerType.NUMBER
    config: Dict[str, Any] = Field(default_factory=lambda: {
        "min_val": 0.0,
        "max_val": 1.0,
    })
    
    @validator('config')
    def validate_number_range(cls, v):
        if v.get('min_val', 0.0) > v.get('max_val', 1.0):
            raise ValueError("min_val must be <= max_val")
        return v

class ListConditionerConfig(BaseConditionerConfig):
    type: Literal[ConditionerType.LIST] = ConditionerType.LIST
    
    @validator('config')
    def validate_options(cls, v):
        if 'options' not in v or not v['options']:
            raise ValueError("options must be specified and non-empty")
        return v

class PretransformConditionerConfig(BaseConditionerConfig):
    type: Literal[ConditionerType.PRETRANSFORM] = ConditionerType.PRETRANSFORM
    
    @validator('config')
    def validate_pretransform_config(cls, v):
        if 'sample_rate' not in v:
            raise ValueError("sample_rate must be specified")
        if not v.get('use_model_pretransform', False) and 'pretransform_config' not in v:
            raise ValueError("pretransform_config must be specified if not using model pretransform")
        return v
```

## Usage Examples

### Text Conditioning
```json
{
    "cond_dim": 1024,
    "configs": [
        {
            "id": "prompt",
            "type": "t5",
            "config": {
                "t5_model_name": "google/flan-t5-large",
                "max_length": 256,
                "enable_grad": false
            }
        }
    ]
}
```

### Multi-Modal Conditioning
```json
{
    "cond_dim": 512,
    "configs": [
        {
            "id": "text",
            "type": "t5",
            "config": {
                "t5_model_name": "t5-base",
                "max_length": 128
            }
        },
        {
            "id": "genre",
            "type": "list",
            "config": {
                "options": ["rock", "pop", "jazz", "classical", "electronic"]
            }
        },
        {
            "id": "tempo",
            "type": "number",
            "config": {
                "min_val": 60.0,
                "max_val": 200.0
            }
        },
        {
            "id": "duration",
            "type": "number",
            "config": {
                "min_val": 1.0,
                "max_val": 300.0
            }
        }
    ],
    "default_keys": {
        "prompt": "text"
    }
}
```

### Audio Conditioning
```json
{
    "cond_dim": 768,
    "configs": [
        {
            "id": "reference_audio",
            "type": "pretransform",
            "config": {
                "sample_rate": 48000,
                "use_model_pretransform": true
            }
        },
        {
            "id": "style_audio",
            "type": "clap_audio",
            "config": {
                "clap_ckpt_path": "/path/to/clap_model.pth"
            }
        }
    ]
}
```

### Source Mixing Conditioning
```json
{
    "cond_dim": 256,
    "configs": [
        {
            "id": "source_mix",
            "type": "source_mix",
            "config": {
                "sample_rate": 44100,
                "use_model_pretransform": false,
                "pretransform_config": {
                    "type": "autoencoder",
                    "config": {
                        "autoencoder_ckpt": "/path/to/autoencoder.ckpt"
                    }
                },
                "source_keys": ["vocals", "drums", "bass", "guitar"],
                "allow_null_source": true,
                "source_length": 1024
            }
        }
    ]
}
```

## Dependencies

The conditioning factory depends on:
- `stable_audio_tools.models.conditioners` - Conditioner implementations
- `stable_audio_tools.models.factory` - Pretransform factory (for pretransform conditioners)
- `transformers` - T5 and tokenizer implementations
- `laion_clap` - CLAP model implementations
- Various audio processing libraries

## Migration Notes

### Current Implementation Issues
- Hard-coded conditioner type strings throughout code
- Limited validation of conditioner configurations
- Complex nested configuration structure
- No validation of conditioning ID consistency across models

### Pydantic Migration Benefits
- Type-safe enums for conditioner types
- Proper validation of all configuration parameters
- Validation of unique conditioner IDs
- Better error messages for configuration issues
- Documentation of all parameters and their purposes
- Support for discriminated unions based on conditioner type
``` 