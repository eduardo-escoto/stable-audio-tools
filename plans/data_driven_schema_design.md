# Data-Driven Schema Design Analysis

## Overview

This document provides a comprehensive analysis of how we should structure our Pydantic configuration schemas based on the **actual factory method analysis** and **real configuration files** from the stable-audio-tools codebase. This analysis revises our original a priori schema design with data-driven insights.

## Key Findings

### 1. Our Original Structure Was Too Simplistic

**Original Plan (a priori):**
```
BaseConfig → ModelConfig, DatasetConfig, TrainingConfig, InferenceConfig
```

**Reality (data-driven):**
```
TopLevelConfig → {
    Model configs: 7 distinct types with completely different structures
    Dataset configs: Simple, just 3 fields
    Training configs: Embedded within model configs, not separate
    Complex nested factories: 50+ configuration types
}
```

### 2. The Real Architecture is Factory-Centric

Every configuration follows this pattern:
```python
{
    "type": "some_type",        # Determines which factory to use
    "config": { ... }           # Type-specific configuration
}
```

This appears at **every level**:
- Model level: `model_type` + `model` config
- Pretransform: `type` + `config` 
- Encoder: `type` + `config`
- Bottleneck: `type` + `config`
- Conditioning: `type` + `config`
- And so on...

### 3. Training Configuration is Model-Specific

Looking at the actual configs, training configuration is:
- **Embedded within model configs** (not separate)
- **Highly specialized** per model type
- **Complex nested structure** (optimizers, schedulers, loss configs, demos)

### 4. Dataset Configuration is Simple

Unlike our expectations, dataset configs are very straightforward:
```json
{
    "dataset_type": "audio_dir",
    "datasets": [{"id": "my_audio", "path": "/path/to/audio/"}],
    "random_crop": true
}
```

### 5. Top-Level Configuration is Training-Focused

The `defaults.ini` file shows the top-level config is primarily about training orchestration:
- Batch size, learning rate, checkpointing
- Paths to model and dataset configs
- PyTorch Lightning trainer parameters

## Factory Method Analysis Summary

From our comprehensive factory analysis (`plans/factory_analysis/`):

### 23+ Factory Functions Across 12 Categories
1. **Model Creation**: 8 functions (autoencoder, diffusion, language model, etc.)
2. **Training Components**: 5 functions (optimizers, schedulers, etc.)
3. **Data Processing**: 4 functions (datasets, preprocessing, etc.)
4. **Specialized Components**: 6+ functions (bottlenecks, conditioning, etc.)

### 50+ Configuration Types Supported
- **Model Types**: 7 (autoencoder, hyperencoder, diffusion_uncond, diffusion_cond, etc.)
- **Bottleneck Types**: 10 (tanh, vae, rvq, dac_rvq, fsq, etc.)
- **Pretransform Types**: 6 (autoencoder, wavelet, pqmf, etc.)
- **Conditioner Types**: 10 (t5, clap_text, clap_audio, etc.)
- **Dataset Types**: 3 (audio_dir, pre_encoded, s3/wds)

## Revised Schema Structure Recommendation

Based on the factory analysis, here's the **data-driven** structure we should implement:

### Core Architecture: Factory Pattern + Discriminated Unions

```python
# 1. Base Factory Pattern
class FactoryConfig(BaseModel):
    """Base class for all factory-based configurations"""
    type: str  # Determines factory to use
    config: Dict[str, Any]  # Type-specific configuration

# 2. Experiment Configuration (what train.py uses)
class ExperimentConfig(BaseModel):
    """Main experiment configuration matching defaults.ini structure"""
    # From defaults.ini
    name: str
    project: Optional[str] = None
    batch_size: int = 4
    seed: int = 42
    num_workers: int = 6
    checkpoint_every: int = 10000
    precision: str = "16-mixed"
    # ... other training params
    
    # From config files
    model_config: str  # Path to model JSON
    dataset_config: str  # Path to dataset JSON
    
# 3. Model Configuration (discriminated by model_type)
class ModelConfig(BaseModel):
    """Model configuration with discriminated union by model_type"""
    model_type: Literal["autoencoder", "diffusion_cond", "diffusion_uncond", "lm", ...]
    sample_rate: int
    sample_size: int
    audio_channels: int = 2
    model: Union[
        AutoencoderModelConfig,
        DiffusionCondModelConfig,
        DiffusionUncondModelConfig,
        LanguageModelConfig,
    ] = Field(discriminator='model_type')
    training: Optional[TrainingConfig] = None

# 4. Specialized Model Configs (based on factory analysis)
class AutoencoderModelConfig(BaseModel):
    """Autoencoder model configuration"""
    encoder: EncoderConfig
    decoder: DecoderConfig
    bottleneck: BottleneckConfig
    latent_dim: int
    downsampling_ratio: int
    io_channels: int

class DiffusionCondModelConfig(BaseModel):
    """Conditional diffusion model configuration"""
    pretransform: PretransformConfig
    conditioning: ConditioningConfig
    diffusion: DiffusionConfig
    io_channels: int

# 5. Factory-based Nested Configs
class EncoderConfig(BaseModel):
    """Encoder configuration with factory pattern"""
    type: Literal["dac", "oobleck", "seanet", ...]
    config: Dict[str, Any]  # Type-specific params

class BottleneckConfig(BaseModel):
    """Bottleneck configuration with factory pattern"""
    type: Literal["vae", "rvq", "dac_rvq", "fsq", ...]
    config: Optional[Dict[str, Any]] = None

class PretransformConfig(BaseModel):
    """Pretransform configuration with factory pattern"""
    type: Literal["autoencoder", "wavelet", "pqmf", ...]
    iterate_batch: bool = True
    config: Dict[str, Any]

class ConditioningConfig(BaseModel):
    """Conditioning configuration with multiple conditioners"""
    configs: List[ConditionerConfig]
    cond_dim: int

class ConditionerConfig(BaseModel):
    """Individual conditioner configuration"""
    id: str
    type: Literal["clap_text", "clap_audio", "t5", "int", "number", ...]
    config: Dict[str, Any]

# 6. Simple Dataset Config
class DatasetConfig(BaseModel):
    """Dataset configuration (simpler than expected)"""
    dataset_type: Literal["audio_dir", "pre_encoded", "s3"]
    datasets: List[DatasetEntry]
    random_crop: bool = True

class DatasetEntry(BaseModel):
    """Individual dataset entry"""
    id: str
    path: str
    custom_metadata_module: Optional[str] = None
```

## Revised Directory Structure

```
stable_audio_tools/config/
├── __init__.py
├── schemas/                    # ← Core Pydantic schemas
│   ├── __init__.py
│   ├── base.py                # Base classes, enums, FactoryConfig
│   ├── experiment.py          # Experiment config (train.py)
│   ├── model.py               # Model configurations
│   ├── dataset.py             # Dataset configurations
│   ├── training.py            # Training configurations
│   ├── factories/             # Factory-specific schemas
│   │   ├── __init__.py
│   │   ├── autoencoder.py
│   │   ├── diffusion.py
│   │   ├── conditioning.py
│   │   ├── pretransform.py
│   │   ├── bottleneck.py
│   │   ├── encoder_decoder.py
│   │   └── loss.py
├── hydra/                     # Hydra configurations
│   ├── config.yaml
│   ├── model/
│   ├── dataset/
│   └── training/
└── validation/                # Validation utilities
    ├── __init__.py
    └── validators.py
```

## Implementation Strategy

### Phase 1: Start with Factory Pattern Foundation
1. **Create base factory classes** that match the `type` + `config` pattern
2. **Implement discriminated unions** for each major factory type
3. **Start with one model type** (e.g., autoencoder) and get it working end-to-end

### Phase 2: Iterative Expansion
1. **Add one model type at a time** (diffusion_cond, diffusion_uncond, etc.)
2. **Add factory-specific schemas** as we encounter them
3. **Preserve backward compatibility** during the transition

### Phase 3: Integration
1. **Migrate train.py** to use the new schemas
2. **Create migration tools** to convert JSON configs to new format
3. **Update factory methods** to accept Pydantic models

## Key Insights

1. **The factory pattern is central** - Our schemas should mirror this exactly
2. **Discriminated unions are essential** - We need them at every level
3. **Start small and iterate** - Don't try to model everything at once
4. **Training configs are embedded** - Not separate as we originally thought
5. **Dataset configs are simple** - Much simpler than we expected
6. **Type safety at every level** - Every factory dispatch needs type validation

## Migration Challenges

### 1. Complex Nested Structures
- Real configs have 4-5 levels of nesting
- Each level uses the factory pattern
- Need to preserve all existing functionality

### 2. Dynamic Configuration Updates
- Current system uses dictionary merging
- Need to maintain this flexibility with Pydantic

### 3. Backward Compatibility
- Existing JSON configs must continue working
- Migration tools needed for smooth transition

### 4. Factory Method Integration
- 23+ factory functions need to work with new schemas
- Type validation at every factory dispatch point

## Specific Examples from Real Configs

### Autoencoder Model Config
```json
{
    "model_type": "autoencoder",
    "sample_rate": 44100,
    "sample_size": 65536,
    "audio_channels": 2,
    "model": {
        "encoder": {
            "type": "dac",
            "config": {
                "in_channels": 2,
                "latent_dim": 128,
                "d_model": 128,
                "strides": [4, 4, 8, 8]
            }
        },
        "decoder": {
            "type": "dac",
            "config": {
                "out_channels": 2,
                "latent_dim": 64,
                "channels": 1536,
                "rates": [8, 8, 4, 4]
            }
        },
        "bottleneck": {
            "type": "vae"
        },
        "latent_dim": 64,
        "downsampling_ratio": 1024,
        "io_channels": 2
    }
}
```

### Diffusion Model Config
```json
{
    "model_type": "diffusion_cond",
    "sample_rate": 44100,
    "sample_size": 4194304,
    "audio_channels": 2,
    "model": {
        "pretransform": {
            "type": "autoencoder",
            "iterate_batch": true,
            "config": {
                "encoder": { "type": "dac", "config": {...} },
                "decoder": { "type": "dac", "config": {...} },
                "bottleneck": { "type": "vae" }
            }
        },
        "conditioning": {
            "configs": [
                {
                    "id": "prompt",
                    "type": "clap_text",
                    "config": {...}
                },
                {
                    "id": "seconds_start",
                    "type": "int",
                    "config": {...}
                }
            ],
            "cond_dim": 768
        },
        "diffusion": {
            "type": "adp_cfg_1d",
            "config": {...}
        }
    }
}
```

## Recommendations

### 1. Start with Base Factory Pattern
- Create `FactoryConfig` base class first
- Implement discriminated unions for major types
- Test with one simple model type

### 2. Use Incremental Approach
- Don't try to model everything at once
- Start with autoencoder (simplest structure)
- Add complexity gradually

### 3. Maintain Flexibility
- Use `Dict[str, Any]` for factory-specific configs initially
- Refine to specific schemas as we learn more
- Keep migration path simple

### 4. Focus on Type Safety
- Use discriminated unions everywhere
- Validate factory dispatch at every level
- Provide clear error messages

## Next Steps

1. **✅ Get feedback** on this analysis and approach - COMPLETE
2. **Create ExperimentConfig** for entry points (train.py and pre_encode.py)
3. **Create base factory classes** to support the experiment config
4. **Implement basic ModelConfig and DatasetConfig** to reference existing JSON files
5. **Set up Hydra integration** for single-file experiment configuration
6. **Create migration tools** from old format to new format

## Questions for Review

1. Does this factory-centric approach make sense?
    - I think it makes sense to me.
2. Should we start with autoencoder or a different model type?
    - Lets start the configs for the entry points of train and pre_encode.
3. Are there any concerns about the discriminated union approach?
    - Can you just summarize what the `discriminitated union` means in the chat?
4. Should we preserve the embedded training configs or separate them?
    - Not sure what you mean by this
5. Any other architectural considerations we should address? 
    - Yeah, I think what might help me is if you can comment on how the downstream hydra configurations might look based on these pydantic datamodels. I think in my end state, it'd be awesome to have the user be able to just use one file for configuring an experiment run that overwrites defaults or specificies how to run the training/pre-encoding. Additionally, ease of use for creating new model configuration defaults. 