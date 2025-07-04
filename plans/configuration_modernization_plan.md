# Configuration System Modernization Plan

## Current Status (Updated)

### ✅ Completed Phases
- **Phase 1.1**: UV Project Migration - Complete migration from setup.py to modern uv project structure
- **Phase 1.3**: Factory Method Analysis - Comprehensive analysis of all 23+ factory functions across 12 categories

### 🚀 Ready to Begin
- **Phase 1.2**: Project Structure - Ready to create the new configuration directory structure
- **Phase 2**: Core Schema Design - Ready to implement Pydantic models based on factory analysis

### 📊 Progress Summary
- **Infrastructure**: 67% complete (2/3 Phase 1 components done)
- **Analysis**: 100% complete (comprehensive factory method documentation)
- **Next Steps**: Create project structure, then begin core schema implementation

### 📁 Deliverables Created
#### Factory Analysis Documentation
- `plans/factory_analysis/` - Complete directory with 12 specialized analysis files
- `plans/factory_methods_analysis.md` - Comprehensive overview and migration strategy
- `plans/training_scripts_analysis/` - Analysis of train.py and pre_encode.py scripts

#### Project Infrastructure
- `pyproject.toml` - Updated with modern dependencies (pydantic, omegaconf, hydra-core)
- `uv.lock` - UV lockfile for reproducible builds
- `plans/uv_migration_plan.md` - Complete migration documentation

### 🎯 Next Steps Quick Reference
1. **Create Project Structure (Phase 1.2)**: Set up the new configuration directory structure
2. **Begin Core Schema Design (Phase 2.1)**: Implement Pydantic models based on factory analysis
3. **Key Resources**: Use `plans/factory_analysis/` for complete parameter specifications
4. **Implementation Guide**: Follow `plans/factory_methods_analysis.md` for migration patterns

---

## Overview

This plan outlines the modernization of the `stable-audio-tools` configuration system from the current brittle prefigure/INI approach to a robust, type-safe system using OmegaConf, Hydra, and Pydantic.

## Current System Analysis

### Issues with Current System
1. **Brittle Configuration Management**: Uses prefigure with INI files that lack structure and validation
2. **Loose JSON Configurations**: Model and dataset configs are unvalidated JSON files
3. **No Schema Validation**: No type checking or validation of configuration values
4. **Poor Developer Experience**: No IDE support, autocomplete, or documentation
5. **Error-Prone**: Runtime errors due to missing/invalid configuration values
6. **Limited Composability**: Hard to compose configurations or override values

### Current Structure
- Entry points: `train.py`, `pre_encode.py`
- Configuration files: `defaults.ini` (for train.py), `pre_encode.py` uses argparse
- Model configs: JSON files in `stable_audio_tools/configs/model_configs/`
- Dataset configs: JSON files in `stable_audio_tools/configs/dataset_configs/`
- Factory pattern: Heavy use of factory methods for object creation
  - **✅ COMPLETED**: Comprehensive analysis of all factory methods (see `plans/factory_analysis/` and `plans/factory_methods_analysis.md`)
  - **Result**: 23+ factory functions analyzed across 12 categories with complete parameter specifications
  - **Goal**: Create composable configuration files that work with existing factory patterns

## Proposed Solution

### Technology Stack
1. **Pydantic**: Data models, validation, and schema generation
2. **OmegaConf**: Configuration management with YAML support
3. **Hydra**: Configuration composition and CLI argument handling
4. **JSON Schema**: Generated schemas for IDE integration

### Key Benefits
1. **Type Safety**: Compile-time and runtime validation
2. **IDE Integration**: Autocomplete, validation, and documentation
3. **Schema Generation**: Automatic JSON schema generation for YAML files
4. **Composability**: Easy configuration composition and overrides
5. **Testability**: Unit tests for configuration validation
6. **Documentation**: Self-documenting configuration structure

## Implementation Plan

### Phase 1: Infrastructure Setup

#### 1.1 UV Project Migration ✅ COMPLETED
- [x] **Complete UV migration** (see [UV Migration Plan](./uv_migration_plan.md) for detailed steps)
  - [x] Migrate from setup.py to modern uv project structure
  - [x] Add new configuration dependencies: `pydantic omegaconf hydra-core`
  - [x] Verify all existing functionality still works
- [x] **Analyze factory methods to understand configuration requirements**
- [ ] Create initial project structure for configuration management

#### 1.2 Project Structure
Create new directory structure:
```
stable_audio_tools/
├── config/
│   ├── __init__.py
│   ├── datamodels/       # Pydantic data models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── training.py
│   │   └── inference.py
│   ├── hydra/            # Hydra configuration files
│   │   ├── config.yaml
│   │   ├── model/
│   │   ├── dataset/
│   │   ├── training/
│   │   └── inference/
│   ├── validation/       # Configuration validation utilities
│   │   ├── __init__.py
│   │   └── validators.py
│   └── migration/        # Migration utilities
│       ├── __init__.py
│       └── converters.py
├── schemas/              # Generated JSON schemas (for IDE integration)
│   ├── model_config_schema.json
│   ├── dataset_config_schema.json
│   └── training_config_schema.json
tests/
├── config/
│   ├── __init__.py
│   ├── test_datamodels.py
│   ├── test_validation.py
│   └── fixtures/
│       ├── valid_configs/
│       └── invalid_configs/
```

#### 1.3 Factory Method Analysis ✅ COMPLETED
- [x] **Analyze all factory methods in `stable_audio_tools/models/factory.py`**
- [x] **Document required configuration parameters for each factory**
- [x] **Identify common patterns and shared configuration structures**
- [x] **Map current JSON configurations to required Pydantic models**

**Results**: Comprehensive analysis completed in `plans/factory_analysis/` directory:
- **23+ factory functions** analyzed across 12 categories
- **50+ configuration types** documented with complete parameter specifications
- **Complete Pydantic schema designs** provided for all factory methods
- **Migration strategies** defined for each factory type
- **Comprehensive overview** in `plans/factory_methods_analysis.md`

### Phase 2: Core Schema Design

#### 2.1 Base Configuration Models
Create foundational Pydantic models:
- [ ] `BaseConfig`: Common configuration base class
- [ ] `ModelConfig`: Model configuration schema
- [ ] `DatasetConfig`: Dataset configuration schema
- [ ] `TrainingConfig`: Training configuration schema
- [ ] `InferenceConfig`: Inference configuration schema

#### 2.2 Model-Specific Schemas
- [ ] `AutoencoderConfig`: Autoencoder model configuration
- [ ] `DiffusionConfig`: Diffusion model configuration
- [ ] `LanguageModelConfig`: Language model configuration
- [ ] `ConditionalConfig`: Conditioning configuration
- [ ] `PreprocessingConfig`: Preprocessing configuration

#### 2.3 Dataset Schemas
- [ ] `AudioDatasetConfig`: Audio dataset configuration
- [ ] `WebDatasetConfig`: Web dataset configuration
- [ ] `LocalDatasetConfig`: Local dataset configuration
- [ ] `S3DatasetConfig`: S3 dataset configuration

### Phase 3: Hydra Integration

#### 3.1 Hydra Configuration Structure
- [ ] Create base Hydra configuration files
- [ ] Setup configuration groups for models, datasets, training
- [ ] Define composition patterns and defaults
- [ ] Create override mechanisms

#### 3.2 Entry Point Modernization
- [ ] Modernize `train.py` with Hydra (migrate from defaults.ini)
- [ ] Modernize `pre_encode.py` with Hydra (migrate from argparse)
- [ ] Create new `train_hydra.py` and `pre_encode_hydra.py`
- [ ] Generate default configuration files for pre_encode workflow
- [ ] Maintain backward compatibility during transition

### Phase 4: Validation and Testing

#### 4.1 Configuration Validation
- [ ] Implement comprehensive validation rules
- [ ] Add cross-field validation
- [ ] Create validation utilities
- [ ] Add validation error reporting

#### 4.2 Testing Framework
- [ ] Create unit tests for all configuration schemas
- [ ] Add integration tests for configuration loading
- [ ] Create fixture-based test data
- [ ] Add property-based testing for edge cases

#### 4.3 Build Hook Implementation
- [ ] Create `build_hooks.py` with schema generation from Pydantic models
- [ ] Add pre-commit hook for schema validation
- [ ] Setup CI/CD validation pipeline
- [ ] Generate JSON schemas for IDE integration

### Phase 5: Migration and Tooling

#### 5.1 Migration Tools
- [ ] Create INI to YAML conversion utility (for train.py)
- [ ] Manually create Hydra configuration for pre_encode.py (examine argparse options)
- [ ] Create JSON to Pydantic model converter
- [ ] Add configuration migration CLI
- [ ] Create validation utility for existing configs

#### 5.2 Developer Tools
- [ ] Add IDE integration documentation
- [ ] Create configuration templates
- [ ] Add configuration validation pre-commit hooks
- [ ] Create configuration debugging tools

## Implementation Details

### 1. Pydantic Schema Example

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum

class ModelType(str, Enum):
    AUTOENCODER = "autoencoder"
    DIFFUSION_COND = "diffusion_cond"
    DIFFUSION_UNCOND = "diffusion_uncond"
    LANGUAGE_MODEL = "lm"

class BaseConfig(BaseModel):
    """Base configuration with common fields"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
class ModelConfig(BaseConfig):
    """Model configuration schema"""
    model_type: ModelType
    sample_rate: int = Field(gt=0, description="Audio sample rate")
    sample_size: int = Field(gt=0, description="Audio sample size")
    audio_channels: int = Field(ge=1, le=8, description="Number of audio channels")
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        if v not in [16000, 22050, 44100, 48000, 96000]:
            raise ValueError(f"Unsupported sample rate: {v}")
        return v
```

### 2. Hydra Configuration Example

```yaml
# config/hydra/config.yaml
defaults:
  - model: stable_audio_1_0
  - dataset: local_training
  - training: default
  - _self_

name: stable_audio_tools
project: null
batch_size: 4
seed: 42
```

### 3. Build Hook Example

```python
# build_hooks.py
import json
from pathlib import Path
from stable_audio_tools.config.datamodels import ModelConfig, DatasetConfig

def generate_schemas():
    """Generate JSON schemas from Pydantic models"""
    schemas_dir = Path("schemas")
    schemas_dir.mkdir(exist_ok=True)
    
    # Generate model config schema
    model_schema = ModelConfig.model_json_schema()
    with open(schemas_dir / "model_config_schema.json", "w") as f:
        json.dump(model_schema, f, indent=2)
    
    # Generate dataset config schema
    dataset_schema = DatasetConfig.model_json_schema()
    with open(schemas_dir / "dataset_config_schema.json", "w") as f:
        json.dump(dataset_schema, f, indent=2)
```

### 4. Testing Framework Example

```python
# tests/config/test_datamodels.py
import pytest
from stable_audio_tools.config.datamodels import ModelConfig, ModelType

def test_model_config_validation():
    """Test model configuration validation"""
    # Valid configuration
    valid_config = {
        "model_type": "autoencoder",
        "sample_rate": 44100,
        "sample_size": 1024,
        "audio_channels": 2
    }
    config = ModelConfig(**valid_config)
    assert config.model_type == ModelType.AUTOENCODER
    
    # Invalid configuration
    with pytest.raises(ValueError):
        ModelConfig(
            model_type="invalid_type",
            sample_rate=44100,
            sample_size=1024,
            audio_channels=2
        )
```

## Risk Assessment

### Technical Risks
1. **Breaking Changes**: Modernization may break existing workflows
   - Mitigation: Maintain backward compatibility during transition
2. **Complex Validation**: Some configurations may be complex to validate
   - Mitigation: Incremental validation implementation
3. **Performance**: Validation overhead may impact performance
   - Mitigation: Optimize validation and add caching

### Project Risks
1. **Scope Creep**: Feature requests may expand scope
   - Mitigation: Stick to core infrastructure first
2. **Adoption**: Team may resist new configuration system
   - Mitigation: Provide clear migration path and documentation

## Success Metrics

### Technical Metrics
- [ ] 100% of existing configurations can be migrated
- [ ] All factory methods work with new configuration system
- [ ] Test coverage > 90% for configuration code
- [ ] Schema validation catches all configuration errors

### Developer Experience Metrics
- [ ] IDE autocomplete works for all configuration files
- [ ] Configuration errors are clear and actionable
- [ ] New configurations can be created without documentation
- [ ] Migration from old system takes < 1 hour

## Post-Implementation

### Next Steps
1. **Factory Method Modernization**: Update all factory methods to use Pydantic models
2. **Advanced Validation**: Add cross-model validation and dependency checking
3. **Configuration UI**: Create web-based configuration editor
4. **Documentation**: Create comprehensive configuration documentation

### Maintenance
1. **Schema Evolution**: Process for updating schemas without breaking changes
2. **Version Compatibility**: Maintain compatibility across versions
3. **Performance Monitoring**: Monitor configuration loading performance
4. **User Feedback**: Collect and incorporate user feedback

## Conclusion

This modernization plan provides a comprehensive approach to replacing the brittle prefigure/INI system with a robust, type-safe configuration management system. The phased approach ensures minimal disruption while maximizing the benefits of modern configuration management tools.

The implementation will significantly improve developer experience, reduce configuration-related bugs, and provide a solid foundation for future enhancements to the stable-audio-tools codebase. 