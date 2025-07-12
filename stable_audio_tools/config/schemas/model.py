"""
Model configuration for deep learning model creation and training.

This module provides model configuration schemas that replace the old JSON-based
model configuration system with type-safe Pydantic models.
"""

from typing import Dict, Any, Optional
from pydantic import Field, ConfigDict
from stable_audio_tools.config.schemas.base import BaseConfig, ModelType


class TrainingConfig(BaseConfig):
    """
    Training configuration for deep learning models.
    
    This is a flexible container for training parameters that vary
    significantly between different model types.
    """
    
    # Common training parameters
    learning_rate: Optional[float] = Field(None, gt=0.0, description="Learning rate for training")
    warmup_steps: Optional[int] = Field(None, ge=0, description="Number of warmup steps")
    use_ema: Optional[bool] = Field(None, description="Whether to use exponential moving averages")
    
    # Flexible container for additional training config
    # We'll make this more specific as we encounter more training configurations
    optimizer_configs: Optional[Dict[str, Any]] = Field(None, description="Optimizer configurations")
    loss_configs: Optional[Dict[str, Any]] = Field(None, description="Loss configurations")
    demo: Optional[Dict[str, Any]] = Field(None, description="Demo/validation configurations")
    
    # Allow additional fields for now since training configs vary significantly
    model_config = ConfigDict(extra="allow")


class ModelConfig(BaseConfig):
    """
    Model configuration for deep learning model creation.
    
    This class provides type-safe configuration for model creation,
    supporting all model types in the stable-audio-tools framework.
    """
    
    # Common fields across all model types
    model_type: ModelType = Field(..., description="Type of model to create")
    sample_rate: int = Field(..., gt=0, description="Audio sample rate in Hz")
    sample_size: int = Field(..., gt=0, description="Audio sample size for training")
    audio_channels: int = Field(2, ge=1, le=16, description="Number of audio channels")
    
    # Model-specific configuration - flexible for now
    # TODO: Replace with discriminated union based on model_type
    model: Dict[str, Any] = Field(..., description="Model-specific configuration")
    
    # Optional training configuration
    training: Optional[TrainingConfig] = Field(None, description="Training configuration")
    
    def get_model_specific_config(self) -> Dict[str, Any]:
        """Get the model-specific configuration dictionary."""
        return self.model
    
    def has_training_config(self) -> bool:
        """Check if training configuration is present."""
        return self.training is not None
    
    def get_training_learning_rate(self) -> Optional[float]:
        """Get the learning rate from training config if present."""
        if self.training:
            return self.training.learning_rate
        return None
    
    def is_autoencoder(self) -> bool:
        """Check if this is an autoencoder model."""
        return self.model_type in [ModelType.AUTOENCODER, ModelType.HYPERENCODER, ModelType.DIFFUSION_AUTOENCODER]
    
    def is_diffusion(self) -> bool:
        """Check if this is a diffusion model."""
        return self.model_type in [
            ModelType.DIFFUSION_UNCOND, 
            ModelType.DIFFUSION_COND, 
            ModelType.DIFFUSION_COND_INPAINT,
            ModelType.DIFFUSION_AUTOENCODER
        ]
    
    def is_language_model(self) -> bool:
        """Check if this is a language model."""
        return self.model_type == ModelType.LANGUAGE_MODEL 