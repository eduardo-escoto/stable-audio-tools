# Language Model Factory Analysis

## Overview

The language model factory in `stable_audio_tools/models/lm.py` creates audio language models for autoregressive generation. This factory builds complete language models with discrete audio tokenization, conditioning, and pattern-based generation.

## Factory Function

### `create_audio_lm_from_config(config)`

**Purpose**: Creates audio language models from configuration
**Location**: `stable_audio_tools/models/lm.py:471`

## Configuration Schema

### Top-Level Structure
```python
{
    "model": dict,  # Required - model configuration
    "sample_rate": int,  # Required - audio sample rate
}
```

### Model Configuration Structure
```python
{
    "model": {
        "lm": dict,  # Required - language model configuration
        "pretransform": dict,  # Required - pretransform configuration
        "conditioning": dict,  # Optional - conditioning configuration
    }
}
```

### Language Model Configuration
```python
{
    "lm": {
        "type": str,  # Required - language model type
        "config": dict,  # Required - model-specific configuration
        "codebook_pattern": str,  # Optional - codebook pattern type
        "cross_attention_cond_ids": list,  # Optional - cross attention conditioning IDs
        "prepend_cond_ids": list,  # Optional - prepend conditioning IDs
        "global_cond_ids": list,  # Optional - global conditioning IDs
    }
}
```

## Configuration Fields

### Top-Level Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `model` | `dict` | Model configuration | **Required** | Must contain lm and pretransform |
| `sample_rate` | `int` | Audio sample rate | **Required** | Must be positive integer |

### Model Section Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `lm` | `dict` | Language model configuration | **Required** | Must contain type and config |
| `pretransform` | `dict` | Pretransform configuration | **Required** | Must be discrete pretransform |
| `conditioning` | `dict` | Conditioning configuration | `None` | Optional - see conditioning factory |

### Language Model Section Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `type` | `str` | Language model type | **Required** | Must be supported type |
| `config` | `dict` | Model-specific configuration | **Required** | Structure depends on type |
| `codebook_pattern` | `str` | Codebook pattern type | `"delay"` | Must be supported pattern |
| `cross_attention_cond_ids` | `list` | Cross attention conditioning IDs | `[]` | List of strings |
| `prepend_cond_ids` | `list` | Prepend conditioning IDs | `[]` | List of strings |
| `global_cond_ids` | `list` | Global conditioning IDs | `[]` | List of strings |

## Supported Language Model Types

### 1. `continuous_transformer` - Continuous Transformer
- **Class**: `ContinuousTransformerAudioLMBackbone`
- **Purpose**: Transformer backbone for audio language modeling
- **Module**: `stable_audio_tools.models.lm_backbone`

#### Configuration
```python
{
    "type": "continuous_transformer",
    "config": {
        # ContinuousTransformerAudioLMBackbone parameters
        "d_model": int,  # Model dimension
        "n_heads": int,  # Number of attention heads
        "n_layers": int,  # Number of layers
        "causal": bool,  # Whether to use causal attention
        # Additional transformer parameters
    }
}
```

## Supported Codebook Patterns

### Pattern Types

| Pattern | Class | Description |
|---------|-------|-------------|
| `parallel` | `ParallelPatternProvider` | Parallel codebook processing |
| `delay` | `DelayedPatternProvider` | Delayed codebook processing |
| `unroll` | `UnrolledPatternProvider` | Unrolled codebook processing |
| `musiclm` | `MusicLMPattern` | MusicLM-style pattern |

### Default Pattern
- **Default**: `"delay"`
- **Purpose**: Sequential processing with delay between quantizers

## Error Handling

### Configuration Validation Errors
- **Missing model config**: `AssertionError` - "model config must be specified in config"
- **Missing sample_rate**: `AssertionError` - "Must specify sample_rate in config"
- **Missing lm config**: `AssertionError` - "lm config must be specified in model config"
- **Missing lm type**: `AssertionError` - "Must specify lm type in lm config"
- **Missing lm model config**: `AssertionError` - "Must specify lm model config in lm config"

### Model Validation Errors
- **Non-discrete pretransform**: `AssertionError` - "Pretransform must be discrete"
- **Unknown lm type**: `NotImplementedError` - "Unrecognized lm type {lm_type}"

## Special Processing

### Pretransform Requirements
1. **Discrete Requirement**: Pretransform must be discrete (tokenized)
2. **Quantizer Information**: Extracted `num_quantizers` and `codebook_size`
3. **Minimum Input Length**: Set to `pretransform.downsampling_ratio`

### Pattern Provider Setup
1. **Pattern Selection**: Based on `codebook_pattern` configuration
2. **Quantizer Count**: Set to `pretransform.num_quantizers`
3. **Pattern Application**: Applied to language model for token generation

### Conditioning Setup
1. **Multi-Conditioner**: Created from `conditioning` configuration if provided
2. **Conditioning IDs**: Extracted from language model configuration
3. **Pretransform Integration**: Passed to conditioner for proper tokenization

## Return Type

### Language Model Wrapper
- **Class**: `AudioLanguageModelWrapper`
- **Module**: `stable_audio_tools.models.lm`
- **Purpose**: Complete wrapper for audio language model with conditioning

## Architecture Components

### Core Components
1. **Pretransform**: Discrete audio tokenizer
2. **Pattern Provider**: Codebook pattern handling
3. **Backbone**: Language model backbone (e.g., transformer)
4. **Language Model**: Core audio language model
5. **Conditioner**: Multi-modal conditioning (optional)
6. **Wrapper**: Complete model wrapper

### Data Flow
1. Audio → Pretransform → Discrete Tokens
2. Tokens → Pattern Provider → Sequence
3. Sequence → Language Model → Logits
4. Logits → Sampling → Next Tokens
5. Tokens → Pretransform → Audio

## Pydantic Schema Design

### Base Configuration

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum

class LanguageModelType(str, Enum):
    CONTINUOUS_TRANSFORMER = "continuous_transformer"

class CodebookPattern(str, Enum):
    PARALLEL = "parallel"
    DELAY = "delay"
    UNROLL = "unroll"
    MUSICLM = "musiclm"

class LanguageModelConfig(BaseModel):
    """Configuration for language model"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    type: LanguageModelType = Field(..., description="Language model type")
    config: Dict[str, Any] = Field(..., description="Model-specific configuration")
    codebook_pattern: CodebookPattern = Field(CodebookPattern.DELAY, description="Codebook pattern type")
    cross_attention_cond_ids: List[str] = Field(default_factory=list, description="Cross attention conditioning IDs")
    prepend_cond_ids: List[str] = Field(default_factory=list, description="Prepend conditioning IDs")
    global_cond_ids: List[str] = Field(default_factory=list, description="Global conditioning IDs")

class AudioLanguageModelConfig(BaseModel):
    """Configuration for audio language model"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    lm: LanguageModelConfig = Field(..., description="Language model configuration")
    pretransform: Dict[str, Any] = Field(..., description="Pretransform configuration")
    conditioning: Optional[Dict[str, Any]] = Field(None, description="Conditioning configuration")

class AudioLanguageModelConfigContainer(BaseModel):
    """Container for audio language model configuration"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    model: AudioLanguageModelConfig = Field(..., description="Model configuration")
    sample_rate: int = Field(..., description="Audio sample rate", gt=0)
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        if v <= 0:
            raise ValueError("Sample rate must be positive")
        return v
```

### Specific Model Configs

```python
class ContinuousTransformerConfig(BaseModel):
    """Configuration for continuous transformer backbone"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    d_model: int = Field(..., description="Model dimension", gt=0)
    n_heads: int = Field(..., description="Number of attention heads", gt=0)
    n_layers: int = Field(..., description="Number of layers", gt=0)
    causal: bool = Field(True, description="Whether to use causal attention")
    
    @validator('n_heads')
    def validate_heads_divisible(cls, v, values):
        if 'd_model' in values and values['d_model'] % v != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return v

class ContinuousTransformerLanguageModelConfig(LanguageModelConfig):
    type: Literal[LanguageModelType.CONTINUOUS_TRANSFORMER] = LanguageModelType.CONTINUOUS_TRANSFORMER
    config: ContinuousTransformerConfig = Field(..., description="Continuous transformer configuration")
```

## Usage Examples

### Basic Language Model
```json
{
    "model": {
        "lm": {
            "type": "continuous_transformer",
            "config": {
                "d_model": 1024,
                "n_heads": 16,
                "n_layers": 24,
                "causal": true
            },
            "codebook_pattern": "delay"
        },
        "pretransform": {
            "type": "dac_pretrained",
            "config": {
                "model_path": "/path/to/dac_model.pth"
            }
        }
    },
    "sample_rate": 44100
}
```

### Language Model with Conditioning
```json
{
    "model": {
        "lm": {
            "type": "continuous_transformer",
            "config": {
                "d_model": 1536,
                "n_heads": 24,
                "n_layers": 36,
                "causal": true
            },
            "codebook_pattern": "delay",
            "cross_attention_cond_ids": ["text"],
            "global_cond_ids": ["genre", "tempo"]
        },
        "pretransform": {
            "type": "autoencoder",
            "config": {
                "autoencoder_ckpt": "/path/to/autoencoder.ckpt"
            }
        },
        "conditioning": {
            "configs": [
                {
                    "id": "text",
                    "type": "t5",
                    "config": {
                        "t5_model_name": "google/flan-t5-large"
                    }
                },
                {
                    "id": "genre",
                    "type": "number",
                    "config": {
                        "min_val": 0,
                        "max_val": 10
                    }
                }
            ]
        }
    },
    "sample_rate": 48000
}
```

### Advanced Pattern Configuration
```json
{
    "model": {
        "lm": {
            "type": "continuous_transformer",
            "config": {
                "d_model": 2048,
                "n_heads": 32,
                "n_layers": 48,
                "causal": true,
                "use_generation_cache": true
            },
            "codebook_pattern": "musiclm",
            "cross_attention_cond_ids": ["text", "audio_context"],
            "prepend_cond_ids": ["artist", "style"],
            "global_cond_ids": ["duration", "key"]
        },
        "pretransform": {
            "type": "dac_pretrained",
            "config": {
                "model_path": "/path/to/dac_model.pth",
                "num_quantizers": 16
            }
        }
    },
    "sample_rate": 44100
}
```

## Dependencies

The language model factory depends on:
- `stable_audio_tools.models.lm` - Language model classes
- `stable_audio_tools.models.lm_backbone` - Backbone implementations
- `stable_audio_tools.models.factory` - Pretransform factory
- `stable_audio_tools.models.conditioners` - Multi-conditioner factory
- `stable_audio_tools.models.codebook_patterns` - Pattern providers

## Migration Notes

### Current Implementation Issues
- Only supports single language model type (`continuous_transformer`)
- Hard-coded pattern provider mapping
- Limited error handling for invalid configurations
- No validation of conditioning ID consistency

### Pydantic Migration Benefits
- Proper validation of all required fields
- Type-safe enums for model types and patterns
- Validation of conditioning ID relationships
- Better error messages for configuration issues
- Documentation of all parameters and their purposes
- Support for multiple language model types through discriminated unions 