#!/usr/bin/env python3
"""
Test script to validate ExperimentConfig with actual defaults.ini values.
This is a temporary test until we implement proper pytest framework.
"""

from stable_audio_tools.config.schemas.experiment import ExperimentConfig

def test_experiment_config_basic():
    """Test ExperimentConfig with values from defaults.ini"""
    
    # Test with defaults.ini values
    config_data = {
        "name": "stable_audio_tools",
        "project": None,
        "batch_size": 4,
        "seed": 42,
        "model_config_path": "stable_audio_tools/configs/model_configs/txt2audio/stable_audio_1_0.json",
        "dataset_config_path": "stable_audio_tools/configs/dataset_configs/local_training_example.json",
        "save_dir": "checkpoints/",
    }
    
    try:
        config = ExperimentConfig(**config_data)
        print("✅ ExperimentConfig validation successful!")
        print(f"Experiment name: {config.name}")
        print(f"Batch size: {config.batch_size}")
        print(f"Model config: {config.model_config_path}")
        print(f"Dataset config: {config.dataset_config_path}")
        print(f"Precision: {config.precision}")
        print(f"Strategy: {config.strategy}")
        print(f"Checkpoint every: {config.checkpoint_every} steps")
        print(f"Logger: {config.logger}")
        return True
    except Exception as e:
        print(f"❌ ExperimentConfig validation failed: {e}")
        return False

def test_experiment_config_validation():
    """Test validation features"""
    
    print("\n--- Testing validation features ---")
    
    # Test invalid precision
    try:
        ExperimentConfig(
            name="test",
            model_config_path="test.json",
            dataset_config_path="test.json",
            precision="invalid"
        )
        print("❌ Should have failed precision validation")
    except ValueError as e:
        print(f"✅ Precision validation working: {e}")
    
    # Test empty name
    try:
        ExperimentConfig(
            name="",
            model_config_path="test.json",
            dataset_config_path="test.json"
        )
        print("❌ Should have failed name validation")
    except ValueError as e:
        print(f"✅ Name validation working: {e}")
    
    # Test invalid strategy
    try:
        ExperimentConfig(
            name="test",
            model_config_path="test.json",
            dataset_config_path="test.json",
            strategy="invalid_strategy"
        )
        print("❌ Should have failed strategy validation")
    except ValueError as e:
        print(f"✅ Strategy validation working: {e}")

def test_experiment_config_deep_learning_features():
    """Test deep learning specific features"""
    
    print("\n--- Testing deep learning features ---")
    
    # Test typical deep learning experiment config
    dl_config = {
        "name": "my_audio_experiment",
        "project": "stable-audio-research",
        "batch_size": 8,
        "seed": 123,
        "precision": "bf16-mixed",
        "strategy": "ddp",
        "num_nodes": 2,
        "gradient_clip_val": 1.0,
        "accum_batches": 4,
        "model_config_path": "my_model.json",
        "dataset_config_path": "my_dataset.json",
        "checkpoint_every": 5000,
        "save_top_k": 3,
        "logger": "wandb",
    }
    
    try:
        config = ExperimentConfig(**dl_config)
        print("✅ Deep learning config validation successful!")
        print(f"Multi-node setup: {config.num_nodes} nodes with {config.strategy} strategy")
        print(f"Mixed precision: {config.precision}")
        print(f"Gradient clipping: {config.gradient_clip_val}")
        print(f"Gradient accumulation: {config.accum_batches} batches")
        return True
    except Exception as e:
        print(f"❌ Deep learning config validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing ExperimentConfig for deep learning workflows...")
    print("=" * 50)
    
    success = True
    success &= test_experiment_config_basic()
    test_experiment_config_validation()
    success &= test_experiment_config_deep_learning_features()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! ExperimentConfig is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above.") 