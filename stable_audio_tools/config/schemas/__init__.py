"""
Pydantic schema definitions for stable-audio-tools configuration.

This package contains all the Pydantic models that define the structure and validation
for configuration files, supporting both the factory pattern and discriminated unions.
"""

from stable_audio_tools.config.schemas.base import FactoryConfig, DatasetType, ModelType
from stable_audio_tools.config.schemas.experiment import ExperimentConfig
from stable_audio_tools.config.schemas.dataset import DatasetConfig, DatasetEntry
from stable_audio_tools.config.schemas.model import ModelConfig, TrainingConfig

__all__ = [
    "FactoryConfig",
    "ExperimentConfig", 
    "DatasetConfig",
    "DatasetEntry",
    "ModelConfig",
    "TrainingConfig",
    "DatasetType",
    "ModelType",
] 