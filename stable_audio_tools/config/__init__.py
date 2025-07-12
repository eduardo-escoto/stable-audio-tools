"""
Configuration system for stable-audio-tools.

This module provides a modern, type-safe configuration system using Pydantic models
and Hydra for configuration management. It replaces the old prefigure/INI-based system.
"""

from stable_audio_tools.config.schemas.experiment import ExperimentConfig
from stable_audio_tools.config.schemas.dataset import DatasetConfig, DatasetEntry
from stable_audio_tools.config.schemas.model import ModelConfig, TrainingConfig
from stable_audio_tools.config.schemas.base import FactoryConfig, DatasetType, ModelType

__all__ = [
    "ExperimentConfig", 
    "DatasetConfig",
    "DatasetEntry",
    "ModelConfig",
    "TrainingConfig",
    "FactoryConfig",
    "DatasetType",
    "ModelType",
] 