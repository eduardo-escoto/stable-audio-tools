# UV Project Migration Plan

## Overview

This document outlines the migration of `stable-audio-tools` from the traditional `setup.py` approach to a modern uv-managed project structure.

## Current State

- **Project Structure**: Uses `setup.py` for dependency management
- **Dependencies**: 28 dependencies with mixed versioning strategies
- **Build System**: setuptools-based build system
- **Package Discovery**: Uses `find_packages()`

## Target State

- **Project Structure**: Modern `pyproject.toml` with uv management
- **Dependencies**: uv-managed dependencies with lock file
- **Build System**: Modern build system configuration
- **Package Discovery**: Explicit package configuration

## Migration Steps

### Step 1: Initialize UV Project
```bash
uv init --name stable-audio-tools
```
This will create a proper `pyproject.toml` structure.

### Step 2: Migrate Project Metadata
Update `pyproject.toml` with project information from `setup.py`:
```toml
[project]
name = "stable-audio-tools"
version = "0.0.19"
description = "Training and inference tools for generative audio models from Stability AI"
authors = [
    {name = "Stability AI"}
]
urls = {Homepage = "https://github.com/Stability-AI/stable-audio-tools.git"}
requires-python = ">=3.8"
```

### Step 3: Analyze Dependencies
Current dependencies from `setup.py` categorized by constraint strategy:

**Unconstrained (let uv choose the best versions):**
```
einops, einops-exts, huggingface_hub, safetensors, tqdm, transformers
alias-free-torch, auraloss, ema-pytorch, importlib-resources, laion-clap
local-attention, pandas, prefigure, PyWavelets, sentencepiece, torchmetrics
v-diffusion-pytorch, vector-quantize-pytorch, wandb, webdataset
```

**Minimum constraints (critical packages only):**
```
torch>=2.5.1              # PyTorch ecosystem
torchaudio>=2.5.1         # PyTorch ecosystem
pytorch_lightning>=2.1.0  # Training framework
gradio>=5.20.0            # UI framework
encodec>=0.1.1            # Audio codec
descript-audio-codec>=1.0.0  # Audio codec
k-diffusion>=0.1.1        # Diffusion models
```

### Step 4: Add Dependencies with UV
Add dependencies with minimal constraints:
```bash
# Let uv choose versions for most packages (unconstrained)
uv add einops einops-exts huggingface_hub safetensors tqdm transformers \
       alias-free-torch auraloss ema-pytorch importlib-resources laion-clap \
       local-attention pandas prefigure PyWavelets sentencepiece torchmetrics \
       v-diffusion-pytorch vector-quantize-pytorch wandb webdataset

# Use minimum constraints for critical packages only
uv add "torch>=2.5.1" "torchaudio>=2.5.1" "pytorch_lightning>=2.1.0" \
       "gradio>=5.20.0" "encodec>=0.1.1" "descript-audio-codec>=1.0.0" \
       "k-diffusion>=0.1.1"
```

**Strategy**: 
- **Unconstrained**: Let uv choose the best versions for most packages
- **Minimum constraints**: Only constrain critical packages (PyTorch ecosystem, audio codecs, diffusion models)

If any compatibility issues arise, we can add upper bounds later (e.g., `"encodec>=0.1.1,<0.2.0"`).

### Step 5: Configure Package Discovery
Update `pyproject.toml` to include proper package discovery:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["stable_audio_tools"]
```

### Step 6: Test Migration
Verify the migration works:
```bash
# Install the package in development mode
uv pip install -e .

# Test import
python -c "import stable_audio_tools; print('Import successful')"

# Test existing functionality
python train.py --help
python pre_encode.py --help
```

### Step 7: Add New Configuration Dependencies
Add our new configuration management dependencies:
```bash
uv add pydantic omegaconf hydra-core jsonschema
```

### Step 8: Clean Up
Remove the old `setup.py` file:
```bash
rm setup.py
```

## Verification Checklist

- [x] `pyproject.toml` contains all project metadata
- [x] All dependencies are properly migrated
- [x] Package can be installed with `uv sync`
- [x] Existing entry points still work (`train.py --help`, `pre_encode.py --help`)
- [x] Import statements function correctly
- [x] New configuration dependencies are available (pydantic, omegaconf, hydra-core)
- [x] `uv.lock` file is created (657KB)
- [x] Old `setup.py` is removed

## ✅ Migration Complete!

## Potential Issues and Solutions

### Issue: Version Conflicts
**Problem**: uv might resolve newer versions that introduce breaking changes
**Solution**: If conflicts arise, add upper bounds to problematic dependencies (e.g., `"package>=1.0.0,<2.0.0"`)

### Issue: Missing Dependencies
**Problem**: Some dependencies might not be recognized
**Solution**: Add them manually with `uv add`

### Issue: Entry Points
**Problem**: Command line scripts might not work
**Solution**: Configure entry points in `pyproject.toml`:
```toml
[project.scripts]
stable-audio-train = "stable_audio_tools.train:main"
stable-audio-pre-encode = "stable_audio_tools.pre_encode:main"
```

### Issue: Build System Changes
**Problem**: Different build backend might cause issues
**Solution**: Test thoroughly and consider using setuptools backend if needed

## Benefits of Migration

1. **Modern Tooling**: Access to uv's fast dependency resolution
2. **Lock Files**: Reproducible builds with `uv.lock`
3. **Better Dependency Management**: Cleaner separation of dev/prod dependencies
4. **PEP 621 Compliance**: Modern Python packaging standards
5. **Faster Installation**: uv's optimized installation process

## Next Steps

After successful migration:
1. Update CI/CD to use uv
2. Add development dependencies using `uv add --dev`
3. Configure additional tools (linting, formatting) in `pyproject.toml`
4. Consider adding optional dependencies for different use cases 