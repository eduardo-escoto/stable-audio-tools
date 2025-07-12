"""
Test Hydra integration with Pydantic validation.
"""

from pathlib import Path
from stable_audio_tools.config import ExperimentConfig, ModelConfig, DatasetConfig

# Test data - this simulates what Hydra would provide
sample_hydra_config = {
    "name": "test_experiment",
    "project": "test_project",
    "batch_size": 4,
    "seed": 42,
    "num_nodes": 1,
    "strategy": "auto",
    "precision": "16-mixed",
    "num_workers": 6,
    "checkpoint_every": 10000,
    "val_every": -1,
    "save_top_k": -1,
    "recover": False,
    "ckpt_path": None,
    "pretrained_ckpt_path": None,
    "pretransform_ckpt_path": None,
    "accum_batches": 1,
    "gradient_clip_val": 0.0,
    "remove_pretransform_weight_norm": False,
    "logger": "wandb",
    "save_dir": "checkpoints/",
    
    # Model configuration
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
            }
        },
        "conditioning": {
            "configs": [
                {
                    "id": "prompt",
                    "type": "clap_text",
                    "config": {
                        "audio_model_type": "HTSAT-base",
                        "enable_fusion": True,
                        "clap_ckpt_path": "/path/to/clap.ckpt",
                        "use_text_features": True,
                        "feature_layer_ix": -2
                    }
                }
            ],
            "cond_dim": 768
        },
        "diffusion": {
            "type": "adp_cfg_1d",
            "cross_attention_cond_ids": ["prompt"],
            "config": {
                "in_channels": 64,
                "context_embedding_features": 768,
                "context_embedding_max_length": 79,
                "channels": 256,
                "resnet_groups": 16,
                "kernel_multiplier_downsample": 2,
                "multipliers": [4, 4, 4, 5, 5],
                "factors": [1, 2, 2, 4],
                "num_blocks": [2, 2, 2, 2],
                "attentions": [1, 3, 3, 3, 3],
                "attention_heads": 16,
                "attention_multiplier": 4,
                "use_nearest_upsample": False,
                "use_skip_scale": True,
                "use_context_time": True
            }
        },
        "io_channels": 64
    },
    "training": {
        "learning_rate": 4e-5,
        "demo": {
            "demo_every": 2000,
            "demo_steps": 250,
            "num_demos": 4,
            "demo_cond": [
                {"prompt": "Test prompt", "seconds_start": 0, "seconds_total": 30}
            ],
            "demo_cfg_scales": [3, 6, 9]
        }
    },
    
    # Dataset configuration
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "test_audio",
            "path": "/path/to/test/audio/",
            "custom_metadata_module": None
        }
    ],
    "random_crop": True
}


def test_hydra_to_pydantic_conversion():
    """Test that Hydra configuration can be converted to Pydantic models."""
    
    # Extract model configuration
    model_dict = {
        "model_type": sample_hydra_config["model_type"],
        "sample_rate": sample_hydra_config["sample_rate"],
        "sample_size": sample_hydra_config["sample_size"],
        "audio_channels": sample_hydra_config["audio_channels"],
        "model": sample_hydra_config["model"],
        "training": sample_hydra_config["training"]
    }
    
    # Extract dataset configuration
    dataset_dict = {
        "dataset_type": sample_hydra_config["dataset_type"],
        "datasets": sample_hydra_config["datasets"],
        "random_crop": sample_hydra_config["random_crop"]
    }
    
    # Extract experiment configuration
    experiment_dict = {
        "name": sample_hydra_config["name"],
        "project": sample_hydra_config["project"],
        "batch_size": sample_hydra_config["batch_size"],
        "seed": sample_hydra_config["seed"],
        "num_nodes": sample_hydra_config["num_nodes"],
        "strategy": sample_hydra_config["strategy"],
        "precision": sample_hydra_config["precision"],
        "num_workers": sample_hydra_config["num_workers"],
        "checkpoint_every": sample_hydra_config["checkpoint_every"],
        "val_every": sample_hydra_config["val_every"],
        "save_top_k": sample_hydra_config["save_top_k"],
        "recover": sample_hydra_config["recover"],
        "ckpt_path": sample_hydra_config["ckpt_path"],
        "pretrained_ckpt_path": sample_hydra_config["pretrained_ckpt_path"],
        "pretransform_ckpt_path": sample_hydra_config["pretransform_ckpt_path"],
        "model_config_path": "hydra://model",
        "dataset_config_path": "hydra://dataset",
        "val_dataset_config": None,
        "accum_batches": sample_hydra_config["accum_batches"],
        "gradient_clip_val": sample_hydra_config["gradient_clip_val"],
        "remove_pretransform_weight_norm": sample_hydra_config["remove_pretransform_weight_norm"],
        "logger": sample_hydra_config["logger"],
        "save_dir": sample_hydra_config["save_dir"]
    }
    
    # Validate all configurations
    model_config = ModelConfig(**model_dict)
    dataset_config = DatasetConfig(**dataset_dict)
    experiment_config = ExperimentConfig(**experiment_dict)
    
    # Test model config properties
    assert model_config.model_type == "diffusion_cond"
    assert model_config.is_diffusion()
    assert model_config.has_training_config()
    assert model_config.sample_rate == 44100
    
    # Test dataset config properties
    assert dataset_config.dataset_type == "audio_dir"
    assert len(dataset_config.datasets) == 1
    assert dataset_config.get_dataset_ids() == ["test_audio"]
    
    # Test experiment config properties
    assert experiment_config.name == "test_experiment"
    assert experiment_config.batch_size == 4
    assert experiment_config.precision == "16-mixed"
    
    print("✅ Hydra to Pydantic conversion test passed!")


def test_hydra_config_files_exist():
    """Test that all Hydra configuration files exist."""
    
    config_dir = Path("stable_audio_tools/config/hydra")
    
    # Check main config
    assert (config_dir / "config.yaml").exists()
    
    # Check model configs
    assert (config_dir / "model" / "stable_audio_1_0.yaml").exists()
    
    # Check dataset configs
    assert (config_dir / "dataset" / "local_training.yaml").exists()
    
    # Check experiment configs
    assert (config_dir / "experiments" / "quick_test.yaml").exists()
    assert (config_dir / "experiments" / "production_run.yaml").exists()
    
    print("✅ All Hydra configuration files exist!")


if __name__ == "__main__":
    test_hydra_to_pydantic_conversion()
    test_hydra_config_files_exist()
    print("🎉 All Hydra integration tests passed!") 