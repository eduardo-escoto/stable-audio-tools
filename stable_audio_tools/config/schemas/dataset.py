"""
Dataset configuration for deep learning data loading workflows.

This module provides dataset configuration schemas that replace the old JSON-based
dataset configuration system with type-safe Pydantic models.
"""

from typing import List, Optional
from pydantic import Field, ConfigDict
from stable_audio_tools.config.schemas.base import BaseConfig, DatasetType


class DatasetEntry(BaseConfig):
    """
    Individual dataset entry configuration.
    
    Represents a single dataset source with its path and optional metadata.
    """
    id: str = Field(..., description="Unique identifier for this dataset")
    path: str = Field(..., description="Path to the dataset (local path or S3 URI)")
    custom_metadata_module: Optional[str] = Field(
        None, 
        description="Path to custom metadata module for this dataset"
    )


class DatasetConfig(BaseConfig):
    """
    Dataset configuration for deep learning data loading.
    
    This class provides type-safe configuration for dataset loading,
    supporting multiple dataset sources and types.
    """
    
    dataset_type: DatasetType = Field(..., description="Type of dataset loading to use")
    datasets: List[DatasetEntry] = Field(..., min_length=1, description="List of dataset sources")
    random_crop: bool = Field(True, description="Whether to randomly crop audio samples")
    
    # Additional fields that may be present in dataset configs
    # We'll add these as we encounter them in real configs
    
    def get_dataset_by_id(self, dataset_id: str) -> Optional[DatasetEntry]:
        """Get a dataset entry by its ID."""
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        return None
    
    def get_dataset_paths(self) -> List[str]:
        """Get all dataset paths."""
        return [dataset.path for dataset in self.datasets]
    
    def get_dataset_ids(self) -> List[str]:
        """Get all dataset IDs."""
        return [dataset.id for dataset in self.datasets] 