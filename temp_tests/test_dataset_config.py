#!/usr/bin/env python3
"""
Test script to validate DatasetConfig with actual dataset configuration values.
This is a temporary test until we implement proper pytest framework.
"""

from stable_audio_tools.config.schemas.dataset import DatasetConfig, DatasetEntry
from stable_audio_tools.config.schemas.base import DatasetType

def test_dataset_config_basic():
    """Test DatasetConfig with values from local_training_example.json"""
    
    # Test with actual local_training_example.json values
    config_data = {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": "my_audio",
                "path": "/path/to/audio/dataset/",
                "custom_metadata_module": "/path/to/custom_metadata/custom_md_example.py"
            }
        ],
        "random_crop": True
    }
    
    try:
        config = DatasetConfig(**config_data)
        print("✅ DatasetConfig validation successful!")
        print(f"Dataset type: {config.dataset_type}")
        print(f"Number of datasets: {len(config.datasets)}")
        print(f"First dataset ID: {config.datasets[0].id}")
        print(f"First dataset path: {config.datasets[0].path}")
        print(f"Custom metadata: {config.datasets[0].custom_metadata_module}")
        print(f"Random crop: {config.random_crop}")
        return True
    except Exception as e:
        print(f"❌ DatasetConfig validation failed: {e}")
        return False

def test_dataset_config_multiple_sources():
    """Test DatasetConfig with multiple dataset sources"""
    
    config_data = {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": "train_data",
                "path": "/path/to/training/audio/",
                "custom_metadata_module": None
            },
            {
                "id": "val_data", 
                "path": "/path/to/validation/audio/",
                "custom_metadata_module": "/custom/validation_metadata.py"
            },
            {
                "id": "test_data",
                "path": "/path/to/test/audio/"
            }
        ],
        "random_crop": False
    }
    
    try:
        config = DatasetConfig(**config_data)
        print("✅ Multiple dataset sources validation successful!")
        print(f"Dataset IDs: {config.get_dataset_ids()}")
        print(f"Dataset paths: {config.get_dataset_paths()}")
        
        # Test helper methods
        train_dataset = config.get_dataset_by_id("train_data")
        print(f"Train dataset path: {train_dataset.path if train_dataset else 'Not found'}")
        
        return True
    except Exception as e:
        print(f"❌ Multiple dataset sources validation failed: {e}")
        return False

def test_dataset_config_s3():
    """Test DatasetConfig with S3 dataset"""
    
    config_data = {
        "dataset_type": "s3",
        "datasets": [
            {
                "id": "s3_audio",
                "path": "s3://my-bucket/audio-data/",
                "custom_metadata_module": None
            }
        ],
        "random_crop": True
    }
    
    try:
        config = DatasetConfig(**config_data)
        print("✅ S3 dataset config validation successful!")
        print(f"S3 path: {config.datasets[0].path}")
        print(f"Dataset type: {config.dataset_type}")
        return True
    except Exception as e:
        print(f"❌ S3 dataset config validation failed: {e}")
        return False

def test_dataset_config_validation():
    """Test validation features"""
    
    print("\n--- Testing validation features ---")
    
    # Test empty datasets list
    try:
        DatasetConfig(
            dataset_type="audio_dir",
            datasets=[],  # Empty list should fail
            random_crop=True
        )
        print("❌ Should have failed with empty datasets list")
    except ValueError as e:
        print(f"✅ Empty datasets validation working: {e}")
    
    # Test invalid dataset type (this won't fail because we use enum, but let's test enum behavior)
    try:
        # This should work fine with valid enum value
        config = DatasetConfig(
            dataset_type=DatasetType.PRE_ENCODED,
            datasets=[{"id": "test", "path": "/test"}],
            random_crop=True
        )
        print(f"✅ Enum validation working: {config.dataset_type}")
    except Exception as e:
        print(f"❌ Enum validation failed: {e}")

if __name__ == "__main__":
    print("Testing DatasetConfig for deep learning data loading...")
    print("=" * 60)
    
    success = True
    success &= test_dataset_config_basic()
    print()
    success &= test_dataset_config_multiple_sources()
    print()
    success &= test_dataset_config_s3()
    test_dataset_config_validation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! DatasetConfig is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above.") 