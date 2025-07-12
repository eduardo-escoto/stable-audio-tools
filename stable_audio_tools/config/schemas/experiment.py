"""
Experiment configuration for deep learning training and pre-encoding workflows.

This module provides the ExperimentConfig class that replaces the old defaults.ini
system with a modern, type-safe configuration approach.
"""

from typing import Optional, Literal
from pydantic import Field, field_validator
from stable_audio_tools.config.schemas.base import BaseConfig


class ExperimentConfig(BaseConfig):
    """
    Main experiment configuration for deep learning workflows.
    
    This class replaces the old defaults.ini system and provides type-safe
    configuration for training and pre-encoding experiments.
    """
    
    # Experiment identification
    name: str = Field(..., description="Name of the experiment run")
    project: Optional[str] = Field(None, description="Project name (e.g., for wandb)")
    
    # Training parameters
    batch_size: int = Field(4, ge=1, description="Training batch size")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    
    # PyTorch Lightning trainer configuration
    num_nodes: int = Field(1, ge=1, description="Number of nodes for distributed training")
    strategy: str = Field("auto", description="Multi-GPU strategy for PyTorch Lightning")
    precision: str = Field("16-mixed", description="Training precision")
    num_workers: int = Field(6, ge=0, description="Number of CPU workers for DataLoader")
    
    # Checkpointing and validation
    checkpoint_every: int = Field(10000, ge=1, description="Steps between checkpoints")
    val_every: int = Field(-1, description="Steps between validation runs (-1 to disable)")
    save_top_k: int = Field(-1, description="Save top K model checkpoints (-1 for all)")
    
    # Resume and recovery
    recover: bool = Field(False, description="Resume training from latest checkpoint")
    ckpt_path: Optional[str] = Field(None, description="Specific checkpoint to resume from")
    pretrained_ckpt_path: Optional[str] = Field(None, description="Pretrained checkpoint to start from")
    pretransform_ckpt_path: Optional[str] = Field(None, description="Pretransform checkpoint if needed")
    
    # Configuration file paths (references to existing JSON configs)
    model_config_path: str = Field(..., description="Path to model configuration file")
    dataset_config_path: str = Field(..., description="Path to dataset configuration file")
    val_dataset_config: Optional[str] = Field(None, description="Path to validation dataset config")
    
    # Training optimization
    accum_batches: int = Field(1, ge=1, description="Gradient accumulation batches")
    gradient_clip_val: float = Field(0.0, ge=0.0, description="Gradient clipping value")
    
    # Model-specific options
    remove_pretransform_weight_norm: bool = Field(False, description="Remove weight norm from pretransform")
    
    # Logging and output
    logger: Literal["wandb", "tensorboard", "none"] = Field("wandb", description="Logger type")
    save_dir: str = Field("", description="Directory to save checkpoints")
    
    # TODO: Add field validators back once we debug the Pydantic v2 syntax
    # @field_validator('precision')
    # @classmethod
    # def validate_precision(cls, v):
    #     """Validate PyTorch Lightning precision settings."""
    #     valid_precisions = ["16-mixed", "bf16-mixed", "32", "64", "16", "bf16"]
    #     if v not in valid_precisions:
    #         raise ValueError(f"Precision must be one of {valid_precisions}, got {v}")
    #     return v
    
    # @field_validator('strategy')
    # @classmethod
    # def validate_strategy(cls, v):
    #     """Validate PyTorch Lightning strategy settings."""
    #     valid_strategies = ["auto", "ddp", "ddp_spawn", "deepspeed", "fsdp"]
    #     if v not in valid_strategies:
    #         raise ValueError(f"Strategy must be one of {valid_strategies}, got {v}")
    #     return v
    
    # @field_validator('model_config_path', 'dataset_config_path')
    # @classmethod
    # def validate_config_paths(cls, v):
    #     """Ensure configuration paths are provided."""
    #     if not v or not v.strip():
    #         raise ValueError("Configuration path cannot be empty")
    #     return v.strip()
    
    # @field_validator('name')
    # @classmethod
    # def validate_name(cls, v):
    #     """Validate experiment name for deep learning workflows."""
    #     if not v or not v.strip():
    #         raise ValueError("Experiment name cannot be empty")
    #     # Remove any potentially problematic characters for file systems
    #     v = v.strip()
    #     if len(v) > 100:
    #         raise ValueError("Experiment name must be less than 100 characters")
    #     return v 