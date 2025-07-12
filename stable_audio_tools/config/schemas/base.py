"""
Base configuration classes and enums for the factory pattern.

This module provides the foundational classes that all other configuration schemas
inherit from, including the factory pattern base class and common enums.
"""

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class FactoryConfig(BaseModel):
    """
    Base class for all factory-based configurations.
    
    This follows the consistent pattern used throughout stable-audio-tools:
    {
        "type": "some_type",        # Determines which factory to use
        "config": { ... }           # Type-specific configuration
    }
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    type: str = Field(..., description="Factory type identifier")
    config: Dict[str, Any] = Field(default_factory=dict, description="Type-specific configuration")


class ModelType(str, Enum):
    """Supported model types for the main model factory."""
    AUTOENCODER = "autoencoder"
    HYPERENCODER = "hyperencoder"
    DIFFUSION_UNCOND = "diffusion_uncond"
    DIFFUSION_COND = "diffusion_cond"
    DIFFUSION_COND_INPAINT = "diffusion_cond_inpaint"
    DIFFUSION_AUTOENCODER = "diffusion_autoencoder"
    LANGUAGE_MODEL = "lm"


class DatasetType(str, Enum):
    """Supported dataset types for the dataset factory."""
    AUDIO_DIR = "audio_dir"
    PRE_ENCODED = "pre_encoded"
    S3_WDS = "s3"


class BaseConfig(BaseModel):
    """
    Base configuration class with common settings.
    
    All configuration classes should inherit from this to ensure consistent
    behavior and validation settings.
    """
    model_config = ConfigDict(
        extra="forbid",           # Reject unknown fields
        validate_assignment=True,  # Validate on assignment
        str_strip_whitespace=True # Strip whitespace from strings
    ) 