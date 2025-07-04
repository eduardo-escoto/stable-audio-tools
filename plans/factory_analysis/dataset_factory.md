# Dataset Factory Analysis

## Overview

The dataset factory in `stable_audio_tools/data/dataset.py` creates PyTorch DataLoaders from configuration. This factory supports multiple dataset types including local audio directories, pre-encoded data, and web datasets from S3 or local WebDataset format.

## Factory Function

### `create_dataloader_from_config(dataset_config, batch_size, sample_size=None, sample_rate=None, audio_channels=2, num_workers=4, shuffle=True)`

**Purpose**: Creates PyTorch DataLoaders from dataset configuration
**Location**: `stable_audio_tools/data/dataset.py:891`

## Function Parameters

| Parameter | Type | Description | Default | Validation |
|-----------|------|-------------|---------|------------|
| `dataset_config` | `dict` | Dataset configuration | **Required** | Must contain dataset_type |
| `batch_size` | `int` | Batch size | **Required** | Must be positive |
| `sample_size` | `int` | Audio sample size | `None` | Optional |
| `sample_rate` | `int` | Audio sample rate | `None` | Optional |
| `audio_channels` | `int` | Number of audio channels | `2` | 1 (mono) or 2 (stereo) |
| `num_workers` | `int` | Number of worker processes | `4` | Must be non-negative |
| `shuffle` | `bool` | Whether to shuffle data | `True` | Boolean |

## Configuration Schema

### Common Structure
```python
{
    "dataset_type": str,  # Required - dataset type identifier
    "datasets": list,  # Required - list of dataset configurations
    "random_crop": bool,  # Optional - whether to use random cropping
    "drop_last": bool,  # Optional - whether to drop last incomplete batch
    # Additional fields depend on dataset type
}
```

### Common Fields

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `dataset_type` | `str` | Dataset type identifier | **Required** | Must be supported type |
| `datasets` | `list` | List of dataset configurations | **Required** | Must be non-empty list |
| `random_crop` | `bool` | Random cropping | `True` | Boolean |
| `drop_last` | `bool` | Drop last incomplete batch | `True` | Boolean |

## Supported Dataset Types

### 1. `audio_dir` - Local Audio Directory
- **Purpose**: Load audio files from local directories
- **Dataset Class**: `SampleDataset`
- **Use Case**: Local training with audio files

#### Configuration
```python
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": str,  # Required - dataset identifier
            "path": str,  # Required - directory path
            "custom_metadata_module": str,  # Optional - custom metadata module path
        }
    ],
    "random_crop": bool,  # Optional - random cropping
    "drop_last": bool,  # Optional - drop last batch
}
```

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `id` | `str` | Dataset identifier | **Required** | Must be unique string |
| `path` | `str` | Directory path | **Required** | Must be valid directory |
| `custom_metadata_module` | `str` | Custom metadata module path | `None` | Optional file path |

#### Features
- **File Types**: Supports .wav, .mp3, .flac, .ogg, .aif, .opus
- **Metadata**: Optional custom metadata extraction
- **Channel Handling**: Automatic mono/stereo conversion
- **Audio Processing**: Automatic resampling and cropping

### 2. `pre_encoded` - Pre-Encoded Data
- **Purpose**: Load pre-encoded latent representations
- **Dataset Class**: `PreEncodedDataset`
- **Use Case**: Training with pre-computed latents

#### Configuration
```python
{
    "dataset_type": "pre_encoded",
    "datasets": [
        {
            "id": str,  # Required - dataset identifier
            "path": str,  # Required - directory path
            "custom_metadata_module": str,  # Optional - custom metadata module path
        }
    ],
    "latent_crop_length": int,  # Optional - latent crop length
    "min_length_sec": float,  # Optional - minimum length in seconds
    "max_length_sec": float,  # Optional - maximum length in seconds
    "random_crop": bool,  # Optional - random cropping
    "extensions": list,  # Optional - file extensions
    "safetensor_key": str,  # Optional - safetensor key
    "drop_last": bool,  # Optional - drop last batch
}
```

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `latent_crop_length` | `int` | Latent crop length | `None` | Optional positive integer |
| `min_length_sec` | `float` | Minimum length in seconds | `None` | Optional positive float |
| `max_length_sec` | `float` | Maximum length in seconds | `None` | Optional positive float |
| `extensions` | `list` | File extensions | `["npy"]` | List of strings |
| `safetensor_key` | `str` | Safetensor key | `"latents"` | String |

#### Features
- **File Formats**: NumPy (.npy), SafeTensors (.safetensors)
- **Length Filtering**: Optional min/max length filtering
- **Metadata**: Optional custom metadata extraction
- **Cropping**: Optional latent-space cropping

### 3. `s3` / `wds` - Web Dataset
- **Purpose**: Load data from S3 or local WebDataset archives
- **Dataset Class**: `WebDatasetDataLoader`
- **Use Case**: Large-scale distributed training

#### Configuration
```python
{
    "dataset_type": "s3",  # or "wds"
    "datasets": [
        {
            "id": str,  # Required - dataset identifier
            "s3_path": str,  # Required for S3 - S3 path
            "path": str,  # Required for local - local path
            "profile": str,  # Optional - AWS profile
            "custom_metadata_module": str,  # Optional - custom metadata module path
        }
    ],
    "remove_silence": bool,  # Optional - remove silence
    "silence_threshold": list,  # Optional - silence threshold
    "max_silence_duration": float,  # Optional - max silence duration
    "random_crop": bool,  # Optional - random cropping
    "volume_norm": bool,  # Optional - volume normalization
    "volume_norm_param": list,  # Optional - volume normalization parameters
    "epoch_steps": int,  # Optional - steps per epoch
    "pre_encoded": bool,  # Optional - pre-encoded data
    "latent_crop_length": int,  # Optional - latent crop length
    "resampled_shards": bool,  # Optional - resample shards
}
```

| Field | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `remove_silence` | `bool` | Remove silence | `False` | Boolean |
| `silence_threshold` | `list` | Silence threshold | `[0.01, 0.5]` | List of floats |
| `max_silence_duration` | `float` | Max silence duration | `0.25` | Positive float |
| `volume_norm` | `bool` | Volume normalization | `False` | Boolean |
| `volume_norm_param` | `list` | Volume norm parameters | `[-16, 2]` | List of numbers |
| `epoch_steps` | `int` | Steps per epoch | `2000` | Positive integer |
| `pre_encoded` | `bool` | Pre-encoded data | `False` | Boolean |
| `resampled_shards` | `bool` | Resample shards | `True` | Boolean |

#### Features
- **Storage**: S3 or local WebDataset archives
- **Streaming**: Efficient streaming from cloud storage
- **Audio Processing**: Silence removal, volume normalization
- **Scalability**: Designed for large-scale datasets
- **Sharding**: Automatic shard resampling

## Error Handling

### Configuration Validation Errors
- **Missing dataset_type**: `AssertionError` - "Dataset type must be specified in dataset config"
- **Missing datasets**: `AssertionError` - "Directory configuration must be specified in datasets"
- **Missing path**: `AssertionError` - "Path must be set for local audio directory configuration"
- **Missing sample_rate (source_mix)**: `AssertionError` - "Sample rate must be specified for source_mix conditioners"

## Special Processing

### Channel Handling
- **Mono (audio_channels=1)**: `force_channels = "mono"`
- **Stereo (audio_channels=2)**: `force_channels = "stereo"`

### Custom Metadata
All dataset types support custom metadata extraction:
1. **Module Loading**: Dynamic import of custom metadata module
2. **Function Extraction**: Extracts `get_custom_metadata` function
3. **Application**: Applied during dataset creation

### DataLoader Configuration
Common DataLoader settings across all types:
- **Persistent Workers**: `persistent_workers=True`
- **Pin Memory**: `pin_memory=True`
- **Collation Function**: `collate_fn=collation_fn`

## Return Type

### DataLoader
- **Class**: `torch.utils.data.DataLoader`
- **Module**: `torch.utils.data`
- **Purpose**: PyTorch DataLoader for training

## Pydantic Schema Design

### Base Configuration

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
from enum import Enum

class DatasetType(str, Enum):
    AUDIO_DIR = "audio_dir"
    PRE_ENCODED = "pre_encoded"
    S3 = "s3"
    WDS = "wds"

class BaseDatasetItemConfig(BaseModel):
    """Base configuration for individual datasets"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    id: str = Field(..., description="Dataset identifier")
    custom_metadata_module: Optional[str] = Field(None, description="Custom metadata module path")

class AudioDirDatasetConfig(BaseDatasetItemConfig):
    """Configuration for audio directory dataset"""
    path: str = Field(..., description="Directory path")

class S3DatasetConfig(BaseDatasetItemConfig):
    """Configuration for S3 dataset"""
    s3_path: str = Field(..., description="S3 path")
    profile: Optional[str] = Field(None, description="AWS profile")

class LocalWebDatasetConfig(BaseDatasetItemConfig):
    """Configuration for local web dataset"""
    path: str = Field(..., description="Local path")

class BaseDatasetConfig(BaseModel):
    """Base configuration for datasets"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    dataset_type: DatasetType = Field(..., description="Dataset type")
    random_crop: bool = Field(True, description="Random cropping")
    drop_last: bool = Field(True, description="Drop last incomplete batch")

class AudioDirDatasetConfigContainer(BaseDatasetConfig):
    """Configuration for audio directory datasets"""
    dataset_type: Literal[DatasetType.AUDIO_DIR] = DatasetType.AUDIO_DIR
    datasets: List[AudioDirDatasetConfig] = Field(..., description="Audio directory configurations")

class PreEncodedDatasetConfigContainer(BaseDatasetConfig):
    """Configuration for pre-encoded datasets"""
    dataset_type: Literal[DatasetType.PRE_ENCODED] = DatasetType.PRE_ENCODED
    datasets: List[AudioDirDatasetConfig] = Field(..., description="Pre-encoded directory configurations")
    latent_crop_length: Optional[int] = Field(None, description="Latent crop length", gt=0)
    min_length_sec: Optional[float] = Field(None, description="Minimum length in seconds", gt=0)
    max_length_sec: Optional[float] = Field(None, description="Maximum length in seconds", gt=0)
    extensions: List[str] = Field(["npy"], description="File extensions")
    safetensor_key: str = Field("latents", description="Safetensor key")
    
    @validator('max_length_sec')
    def validate_length_range(cls, v, values):
        if v is not None and 'min_length_sec' in values and values['min_length_sec'] is not None:
            if v <= values['min_length_sec']:
                raise ValueError("max_length_sec must be greater than min_length_sec")
        return v

class WebDatasetConfigContainer(BaseDatasetConfig):
    """Configuration for web datasets"""
    dataset_type: Union[Literal[DatasetType.S3], Literal[DatasetType.WDS]] = Field(..., description="Web dataset type")
    datasets: List[Union[S3DatasetConfig, LocalWebDatasetConfig]] = Field(..., description="Web dataset configurations")
    remove_silence: bool = Field(False, description="Remove silence")
    silence_threshold: List[float] = Field([0.01, 0.5], description="Silence threshold")
    max_silence_duration: float = Field(0.25, description="Max silence duration", gt=0)
    volume_norm: bool = Field(False, description="Volume normalization")
    volume_norm_param: List[float] = Field([-16, 2], description="Volume normalization parameters")
    epoch_steps: int = Field(2000, description="Steps per epoch", gt=0)
    pre_encoded: bool = Field(False, description="Pre-encoded data")
    latent_crop_length: Optional[int] = Field(None, description="Latent crop length", gt=0)
    resampled_shards: bool = Field(True, description="Resample shards")
```

## Usage Examples

### Local Audio Directory
```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "music_dataset",
            "path": "/path/to/music/files"
        },
        {
            "id": "speech_dataset", 
            "path": "/path/to/speech/files",
            "custom_metadata_module": "/path/to/custom_metadata.py"
        }
    ],
    "random_crop": true,
    "drop_last": true
}
```

### Pre-Encoded Dataset
```json
{
    "dataset_type": "pre_encoded",
    "datasets": [
        {
            "id": "latents_dataset",
            "path": "/path/to/latents"
        }
    ],
    "latent_crop_length": 1024,
    "min_length_sec": 1.0,
    "max_length_sec": 30.0,
    "extensions": ["npy", "safetensors"],
    "safetensor_key": "latents",
    "random_crop": false
}
```

### S3 Web Dataset
```json
{
    "dataset_type": "s3",
    "datasets": [
        {
            "id": "large_audio_dataset",
            "s3_path": "s3://my-bucket/audio-dataset/",
            "profile": "my-aws-profile"
        }
    ],
    "remove_silence": true,
    "silence_threshold": [0.01, 0.5],
    "max_silence_duration": 0.25,
    "volume_norm": true,
    "volume_norm_param": [-16, 2],
    "epoch_steps": 5000,
    "resampled_shards": true
}
```

### Local Web Dataset
```json
{
    "dataset_type": "wds",
    "datasets": [
        {
            "id": "local_webdataset",
            "path": "/path/to/webdataset/shards"
        }
    ],
    "pre_encoded": true,
    "latent_crop_length": 512,
    "epoch_steps": 1000
}
```

## Dependencies

The dataset factory depends on:
- `torch.utils.data` - PyTorch data loading utilities
- `webdataset` - WebDataset library for efficient data streaming
- `torchaudio` - Audio loading and processing
- `numpy` - NumPy array handling
- `safetensors` - SafeTensors format support
- Custom metadata modules (optional)

## Migration Notes

### Current Implementation Issues
- Hard-coded dataset type strings
- Complex nested configuration validation
- Inconsistent parameter handling across dataset types
- Limited validation of file paths and configurations

### Pydantic Migration Benefits
- Type-safe enums for dataset types
- Proper validation of all configuration parameters
- Validation of file paths and dataset consistency
- Better error messages for configuration issues
- Documentation of all parameters and their purposes
- Support for discriminated unions based on dataset type
- Validation of parameter relationships (e.g., min/max length) 