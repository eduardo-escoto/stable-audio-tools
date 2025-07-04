# Training Scripts Analysis Directory

This directory contains comprehensive analysis of the training and preprocessing scripts to inform the Hydra configuration design.

## Analysis Files

### Entry Point Scripts
- [**train_py_analysis.md**](./train_py_analysis.md) - Complete analysis of `train.py` and its configuration requirements
- [**pre_encode_py_analysis.md**](./pre_encode_py_analysis.md) - Complete analysis of `pre_encode.py` and its arguments
- [**defaults_ini_analysis.md**](./defaults_ini_analysis.md) - Analysis of `defaults.ini` configuration structure

### Configuration Files
- [**existing_configs_analysis.md**](./existing_configs_analysis.md) - Analysis of existing JSON configuration files

## Purpose

Each analysis file provides:
- **Command Line Arguments**: All CLI arguments with types, defaults, and descriptions
- **Configuration Dependencies**: Required configuration files and their structure
- **Parameter Validation**: Required validation rules and constraints
- **Usage Patterns**: How the script is intended to be used
- **Migration Path**: How to convert to Hydra configuration

## Usage

These analyses serve as the **source of truth** for creating Hydra configurations that replace the current prefigure/argparse approach while maintaining backward compatibility. 