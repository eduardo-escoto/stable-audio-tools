"""
Shared helpers for scripts/demo.py and scripts/eval.py.
"""

import json
from pathlib import Path

import torch
import torchaudio

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict


# ---------------------------------------------------------------------------
# Default prompts (fallback when no prompts-json or demo_cond in config)
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    # Full-mix — baseline-friendly
    {"prompt": "Upbeat acoustic pop song with strummed guitar, light percussion, and warm vocals.", "seconds_total": 12},
    {"prompt": "Smooth jazz trio: walking upright bass, brushed snare, and muted trumpet melody.", "seconds_total": 12},
    {"prompt": "Driving electronic dance track with a four-on-the-floor kick, synth bass, and arpeggiated leads.", "seconds_total": 12},
    {"prompt": "Ambient soundscape with slow pad swells, subtle reverb, and gentle melodic phrases.", "seconds_total": 12},
    # Stems — ambient/sPDMX-friendly
    {"prompt": "Solo grand piano, soft arpeggios in a dry room, intimate and reflective.", "seconds_total": 12},
    {"prompt": "Fingerpicked steel-string acoustic guitar, light room reverb, clear picking patterns.", "seconds_total": 12},
    {"prompt": "Electric bass guitar, clean tone, steady groove, no other instruments.", "seconds_total": 12},
    {"prompt": "Analog synthesizer lead melody, warm tone, slight vibrato, minimal accompaniment.", "seconds_total": 12},
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_config: dict, ckpt_path: str, device: torch.device):
    """Load a ConditionedDiffusionModelWrapper from a config + checkpoint."""
    model = create_model_from_config(model_config)

    pretrained_ckpt = model_config.get("pretrained_ckpt_path", None)
    if pretrained_ckpt:
        copy_state_dict(model, load_ckpt_state_dict(pretrained_ckpt))

    pretransform_ckpt = model_config.get("pretransform_ckpt_path", None)
    if pretransform_ckpt and model.pretransform is not None:
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt))

    # Load the user-supplied checkpoint on top (overrides any pretrained weights)
    if ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(ckpt_path))

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Audio saving
# ---------------------------------------------------------------------------

def save_audio(audio: torch.Tensor, path: str, sample_rate: int):
    """Save a (channels, samples) float tensor as a 16-bit WAV file."""
    audio = audio.to(torch.float32).cpu()
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio_int16 = (audio * 32767).clamp(-32768, 32767).to(torch.int16)
    torchaudio.save(path, audio_int16, sample_rate)


# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------

def resolve_prompts(prompts_json: str | None, model_config: dict) -> list[dict]:
    """Return a list of conditioning dicts, resolved in priority order:
    1. --prompts-json file
    2. training.demo.demo_cond in model_config
    3. DEFAULT_PROMPTS fallback
    """
    if prompts_json is not None:
        with open(prompts_json) as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {prompts_json}")
        return prompts

    prompts = (
        model_config.get("training", {})
        .get("demo", {})
        .get("demo_cond", None)
    )
    if prompts:
        print(f"Using {len(prompts)} prompts from model config.")
        return prompts

    print(f"No demo_cond in model config. Using {len(DEFAULT_PROMPTS)} built-in default prompts.")
    return DEFAULT_PROMPTS
