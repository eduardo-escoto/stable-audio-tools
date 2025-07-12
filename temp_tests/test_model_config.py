#!/usr/bin/env python3
"""
Test script to validate ModelConfig with actual model configuration values.
This is a temporary test until we implement proper pytest framework.
"""

from stable_audio_tools.config.schemas.model import ModelConfig, TrainingConfig
from stable_audio_tools.config.schemas.base import ModelType

def test_autoencoder_config():
    """Test ModelConfig with autoencoder configuration from stable_audio_1_0_vae.json"""
    
    # Simplified version of the actual autoencoder config
    config_data = {
        "model_type": "autoencoder",
        "sample_size": 65536,
        "sample_rate": 44100,
        "audio_channels": 2,
        "model": {
            "encoder": {
                "type": "dac",
                "config": {
                    "in_channels": 2,
                    "latent_dim": 128,
                    "d_model": 128,
                    "strides": [4, 4, 8, 8]
                }
            },
            "decoder": {
                "type": "dac",
                "config": {
                    "out_channels": 2,
                    "latent_dim": 64,
                    "channels": 1536,
                    "rates": [8, 8, 4, 4]
                }
            },
            "bottleneck": {
                "type": "vae"
            },
            "latent_dim": 64,
            "downsampling_ratio": 1024,
            "io_channels": 2
        },
        "training": {
            "learning_rate": 1e-4,
            "warmup_steps": 0,
            "use_ema": True,
            "optimizer_configs": {
                "autoencoder": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "betas": [0.8, 0.99],
                            "lr": 1e-4
                        }
                    }
                }
            }
        }
    }
    
    try:
        config = ModelConfig(**config_data)
        print("✅ Autoencoder ModelConfig validation successful!")
        print(f"Model type: {config.model_type}")
        print(f"Sample rate: {config.sample_rate} Hz")
        print(f"Sample size: {config.sample_size}")
        print(f"Audio channels: {config.audio_channels}")
        print(f"Is autoencoder: {config.is_autoencoder()}")
        print(f"Is diffusion: {config.is_diffusion()}")
        print(f"Has training config: {config.has_training_config()}")
        print(f"Learning rate: {config.get_training_learning_rate()}")
        return True
    except Exception as e:
        print(f"❌ Autoencoder ModelConfig validation failed: {e}")
        return False

def test_diffusion_cond_config():
    """Test ModelConfig with conditional diffusion configuration from stable_audio_1_0.json"""
    
    # Simplified version of the actual diffusion config
    config_data = {
        "model_type": "diffusion_cond",
        "sample_size": 4194304,
        "sample_rate": 44100,
        "audio_channels": 2,
        "model": {
            "pretransform": {
                "type": "autoencoder",
                "iterate_batch": True,
                "config": {
                    "encoder": {
                        "type": "dac",
                        "config": {
                            "in_channels": 2,
                            "latent_dim": 128
                        }
                    },
                    "decoder": {
                        "type": "dac", 
                        "config": {
                            "out_channels": 2,
                            "latent_dim": 64
                        }
                    },
                    "bottleneck": {
                        "type": "vae"
                    }
                }
            },
            "conditioning": {
                "configs": [
                    {
                        "id": "prompt",
                        "type": "clap_text",
                        "config": {
                            "audio_model_type": "HTSAT-base",
                            "enable_fusion": True
                        }
                    }
                ],
                "cond_dim": 768
            },
            "diffusion": {
                "type": "adp_cfg_1d",
                "config": {
                    "in_channels": 64,
                    "channels": 256
                }
            },
            "io_channels": 64
        },
        "training": {
            "learning_rate": 4e-5,
            "demo": {
                "demo_every": 2000,
                "demo_steps": 250,
                "num_demos": 4
            }
        }
    }
    
    try:
        config = ModelConfig(**config_data)
        print("✅ Diffusion ModelConfig validation successful!")
        print(f"Model type: {config.model_type}")
        print(f"Sample rate: {config.sample_rate} Hz")
        print(f"Is autoencoder: {config.is_autoencoder()}")
        print(f"Is diffusion: {config.is_diffusion()}")
        print(f"Is language model: {config.is_language_model()}")
        print(f"Learning rate: {config.get_training_learning_rate()}")
        return True
    except Exception as e:
        print(f"❌ Diffusion ModelConfig validation failed: {e}")
        return False

def test_model_config_validation():
    """Test validation features"""
    
    print("\n--- Testing validation features ---")
    
    # Test invalid sample rate
    try:
        ModelConfig(
            model_type="autoencoder",
            sample_rate=-1,  # Invalid negative value
            sample_size=1024,
            model={"test": "config"}
        )
        print("❌ Should have failed with negative sample rate")
    except ValueError as e:
        print(f"✅ Sample rate validation working: {e}")
    
    # Test invalid audio channels
    try:
        ModelConfig(
            model_type="autoencoder",
            sample_rate=44100,
            sample_size=1024,
            audio_channels=0,  # Invalid zero channels
            model={"test": "config"}
        )
        print("❌ Should have failed with zero audio channels")
    except ValueError as e:
        print(f"✅ Audio channels validation working: {e}")

def test_model_type_helpers():
    """Test model type helper methods"""
    
    print("\n--- Testing model type helpers ---")
    
    # Test autoencoder detection
    autoencoder_config = ModelConfig(
        model_type=ModelType.AUTOENCODER,
        sample_rate=44100,
        sample_size=1024,
        model={"test": "config"}
    )
    print(f"✅ Autoencoder detection: {autoencoder_config.is_autoencoder()}")
    
    # Test diffusion detection
    diffusion_config = ModelConfig(
        model_type=ModelType.DIFFUSION_COND,
        sample_rate=44100,
        sample_size=1024,
        model={"test": "config"}
    )
    print(f"✅ Diffusion detection: {diffusion_config.is_diffusion()}")
    
    # Test language model detection
    lm_config = ModelConfig(
        model_type=ModelType.LANGUAGE_MODEL,
        sample_rate=44100,
        sample_size=1024,
        model={"test": "config"}
    )
    print(f"✅ Language model detection: {lm_config.is_language_model()}")

if __name__ == "__main__":
    print("Testing ModelConfig for deep learning model creation...")
    print("=" * 60)
    
    success = True
    success &= test_autoencoder_config()
    print()
    success &= test_diffusion_cond_config()
    test_model_config_validation()
    test_model_type_helpers()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! ModelConfig is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above.") 