"""
Hydra integration demo for stable-audio-tools configuration system.

This demonstrates how Hydra configurations can be loaded and validated
with our Pydantic models, enabling single-file experiment configuration.

Usage:
    python demo_config.py
    python demo_config.py --config-name=experiments/quick_test
    python demo_config.py --config-name=experiments/production_run
"""

import sys
from pathlib import Path

# Add stable_audio_tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from stable_audio_tools.config import ExperimentConfig, ModelConfig, DatasetConfig
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: uv add hydra-core omegaconf")
    raise


@hydra.main(version_base=None, config_path="../../stable_audio_tools/config/hydra", config_name="config")
def demo_hydra_integration(cfg: DictConfig) -> None:
    """
    Demonstrate Hydra integration with Pydantic validation.
    
    This function shows how to:
    1. Load configuration with Hydra composition
    2. Validate with Pydantic models
    3. Access configuration safely with type checking
    """
    
    print("🎉 Hydra + Pydantic Integration Demo")
    print("=" * 50)
    
    # Convert DictConfig to dictionary for Pydantic
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    try:
        # Create and validate the experiment configuration
        print("📋 Creating ExperimentConfig...")
        
        # Extract model and dataset configs
        model_dict = {
            "model_type": cfg_dict["model_type"],
            "sample_rate": cfg_dict["sample_rate"],
            "sample_size": cfg_dict["sample_size"],
            "audio_channels": cfg_dict["audio_channels"],
            "model": cfg_dict["model"],
            "training": cfg_dict.get("training")
        }
        
        dataset_dict = {
            "dataset_type": cfg_dict["dataset_type"],
            "datasets": cfg_dict["datasets"],
            "random_crop": cfg_dict["random_crop"]
        }
        
        experiment_dict = {
            "name": cfg_dict["name"],
            "project": cfg_dict.get("project"),
            "batch_size": cfg_dict["batch_size"],
            "seed": cfg_dict["seed"],
            "num_nodes": cfg_dict["num_nodes"],
            "strategy": cfg_dict["strategy"],
            "precision": cfg_dict["precision"],
            "num_workers": cfg_dict["num_workers"],
            "checkpoint_every": cfg_dict["checkpoint_every"],
            "val_every": cfg_dict["val_every"],
            "save_top_k": cfg_dict["save_top_k"],
            "recover": cfg_dict["recover"],
            "ckpt_path": cfg_dict.get("ckpt_path"),
            "pretrained_ckpt_path": cfg_dict.get("pretrained_ckpt_path"),
            "pretransform_ckpt_path": cfg_dict.get("pretransform_ckpt_path"),
            "model_config_path": "hydra://model",  # Placeholder for Hydra model config
            "dataset_config_path": "hydra://dataset",  # Placeholder for Hydra dataset config
            "val_dataset_config": cfg_dict.get("val_dataset_config"),
            "accum_batches": cfg_dict["accum_batches"],
            "gradient_clip_val": cfg_dict["gradient_clip_val"],
            "remove_pretransform_weight_norm": cfg_dict.get("remove_pretransform_weight_norm", False),
            "logger": cfg_dict["logger"],
            "save_dir": cfg_dict["save_dir"]
        }
        
        # Validate configurations
        model_config = ModelConfig(**model_dict)
        dataset_config = DatasetConfig(**dataset_dict)
        experiment_config = ExperimentConfig(**experiment_dict)
        
        print("✅ All configurations validated successfully!")
        print()
        
        # Display configuration summary
        print("📊 Configuration Summary")
        print("-" * 30)
        print(f"Experiment: {experiment_config.name}")
        print(f"Project: {experiment_config.project}")
        print(f"Batch size: {experiment_config.batch_size}")
        print(f"Precision: {experiment_config.precision}")
        print(f"Strategy: {experiment_config.strategy}")
        print()
        
        print(f"Model: {model_config.model_type}")
        print(f"Sample rate: {model_config.sample_rate} Hz")
        print(f"Sample size: {model_config.sample_size}")
        print(f"Audio channels: {model_config.audio_channels}")
        print(f"Is diffusion model: {model_config.is_diffusion()}")
        print(f"Has training config: {model_config.has_training_config()}")
        print()
        
        print(f"Dataset type: {dataset_config.dataset_type}")
        print(f"Number of datasets: {len(dataset_config.datasets)}")
        print(f"Dataset IDs: {dataset_config.get_dataset_ids()}")
        print(f"Random crop: {dataset_config.random_crop}")
        print()
        
        # Show Hydra working directory
        print("📁 Hydra Working Directory")
        print("-" * 30)
        print(f"Output dir: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
        print()
        
        # Show full configuration (for debugging)
        print("🔍 Full Configuration (YAML)")
        print("-" * 30)
        print(OmegaConf.to_yaml(cfg))
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


if __name__ == "__main__":
    # The cfg parameter will be provided by Hydra decorator
    demo_hydra_integration() 