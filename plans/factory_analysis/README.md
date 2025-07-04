# Factory Methods Analysis Directory

This directory contains comprehensive analysis of all factory methods in `stable-audio-tools` to inform the Pydantic schema design.

## Analysis Files

### Core Factory Methods
- [**model_factory.md**](./model_factory.md) - Main model factory dispatch and model type analysis
- [**pretransform_factory.md**](./pretransform_factory.md) - Pretransform factory patterns and configurations
- [**bottleneck_factory.md**](./bottleneck_factory.md) - Bottleneck factory patterns and configurations
- [**encoder_decoder_factory.md**](./encoder_decoder_factory.md) - Encoder/decoder factory patterns and configurations

### Training & Data Factories
- [**training_factory.md**](./training_factory.md) - Training wrapper factory analysis
- [**dataset_factory.md**](./dataset_factory.md) - Dataset factory analysis

## Purpose

Each analysis file provides:
- **Configuration Schema**: Exact structure and required fields
- **Parameter Types**: Python types and validation rules
- **Default Values**: Default parameters and fallback behavior
- **Validation Rules**: Required validations and constraints
- **Usage Examples**: Real configuration examples from the codebase

## Usage

These analyses serve as the **source of truth** for creating Pydantic schemas that match the existing factory method requirements while adding proper validation and type safety. 