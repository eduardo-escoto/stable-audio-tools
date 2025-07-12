# Hydra Integration Examples

This directory contains examples of how to use the Hydra configuration system with stable-audio-tools.

## Demo Configuration

The `demo_config.py` script demonstrates how to:

1. **Load configuration with Hydra composition** - Automatically merge model, dataset, and experiment configs
2. **Validate with Pydantic models** - Type-safe configuration validation
3. **Access configuration safely** - IDE support with type checking

## Usage

Run the examples from the project root:

```bash
# Default configuration
python examples/hydra_integration/demo_config.py

# Quick test experiment
python examples/hydra_integration/demo_config.py --config-name=experiments/quick_test

# Production experiment
python examples/hydra_integration/demo_config.py --config-name=experiments/production_run

# Override individual parameters
python examples/hydra_integration/demo_config.py batch_size=16 name=my_experiment
```

## Configuration Structure

The Hydra configuration system allows you to:

- **Compose configurations** from multiple files
- **Override any parameter** from the command line
- **Create experiment templates** for common use cases
- **Maintain type safety** with Pydantic validation

## Single-File Experiment Configuration

The key achievement is that you can now define complete experiments in a single YAML file:

```yaml
# experiments/my_experiment.yaml
defaults:
  - ../config
  - model: stable_audio_1_0
  - dataset: local_training
  - _self_

name: my_custom_experiment
batch_size: 8
learning_rate: 2e-5

# Override model parameters
sample_size: 2097152

# Override dataset path
datasets:
  - id: my_data
    path: /path/to/my/data/
```

This replaces the need to manage separate `defaults.ini`, model JSON, and dataset JSON files! 