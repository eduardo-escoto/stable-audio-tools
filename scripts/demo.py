"""
demo.py — Generate audio demos from one or more trained diffusion models.

Replicates what the DiffusionCondDemoCallback does during training, but runs
offline from a checkpoint so you can listen to a model after training is done.

Multiple models can be compared side-by-side: each model gets its own
sub-directory under --output-dir, and the same prompts + CFG scales are used
for every model.

Prompts are specified as a JSON file (list of conditioning dicts matching the
`training.demo.demo_cond` format in the model config), read directly from the
first model config's `training.demo.demo_cond` field, or fall back to a
built-in set of default prompts covering full-mix and stem-level descriptions.

Usage — single model, prompts from config:
    python scripts/demo.py \\
        --model-config stable_audio_tools/configs/model_configs/saosmall_ambient_loss.json \\
        --ckpt /deepfreeze/pnlong/ambient/runs/checkpoints/last.ckpt \\
        --output-dir /tmp/demos

Usage — compare two models with custom prompts:
    python scripts/demo.py \\
        --model-config path/to/baseline.json  --ckpt path/to/baseline.ckpt  --model-name baseline \\
        --model-config path/to/ambient.json   --ckpt path/to/ambient.ckpt   --model-name ambient \\
        --prompts-json path/to/prompts.json \\
        --cfg-scale 1 4 7 \\
        --steps 100 \\
        --output-dir /tmp/compare_demos

Prompts JSON format (list of conditioning dicts):
    [
        {"prompt": "Soft piano arpeggios, intimate and reflective.", "seconds_total": 12},
        {"prompt": "Fingerpicked acoustic guitar, folk ballad.", "seconds_total": 12}
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Ensure the repo root is on the path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Ensure scripts/ itself is on the path for _utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_audio_tools.inference.generation import generate_diffusion_cond
from _utils import load_model, resolve_prompts, save_audio


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "demos"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate audio demos from trained diffusion model checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ----- Model specification (repeatable, one entry per model) -----
    parser.add_argument(
        "--model-config", dest="model_configs", action="append", default=[],
        metavar="PATH",
        help="Path to a model JSON config. Repeat for each model to compare.",
    )
    parser.add_argument(
        "--ckpt", dest="ckpts", action="append", default=[],
        metavar="PATH",
        help="Path to a .ckpt or .safetensors checkpoint. Must appear in the "
             "same order as --model-config. Use an empty string to skip loading "
             "a checkpoint and rely solely on the pretrained_ckpt_path in the config.",
    )
    parser.add_argument(
        "--model-name", dest="model_names", action="append", default=[],
        metavar="NAME",
        help="Display name for each model (used as the sub-directory name). "
             "Defaults to the config filename stem.",
    )

    # ----- Prompts -----
    parser.add_argument(
        "--prompts-json", default=None, metavar="PATH",
        help="Path to a JSON file containing a list of conditioning dicts "
             "(e.g. [{\"prompt\": \"...\", \"seconds_total\": 12}, ...]). "
             "If omitted, prompts are read from the first model config's "
             "training.demo.demo_cond field.",
    )

    # ----- Generation settings -----
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of diffusion steps (default: 100).")
    parser.add_argument(
        "--cfg-scale", dest="cfg_scales", type=float, nargs="+", default=None,
        metavar="SCALE",
        help="One or more CFG scales to sweep. Defaults to the demo_cfg_scales "
             "from the first model config, or [7.0] if absent.",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (default: cuda if available).")

    # ----- Output -----
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
                        help=f"Root directory for output WAV files (default: {DEFAULT_OUTPUT_DIR!r}).")

    args = parser.parse_args()

    # ----- Validate inputs -----
    if not args.model_configs:
        parser.error("At least one --model-config must be provided.")

    if args.ckpts and len(args.ckpts) != len(args.model_configs):
        parser.error(
            f"Number of --ckpt arguments ({len(args.ckpts)}) must match "
            f"number of --model-config arguments ({len(args.model_configs)})."
        )

    # Pad missing ckpts / names
    ckpts = args.ckpts or [""] * len(args.model_configs)
    names = list(args.model_names)
    while len(names) < len(args.model_configs):
        idx = len(names)
        names.append(Path(args.model_configs[idx]).stem)

    device = torch.device(args.device)

    # ----- Load all model configs -----
    model_configs = []
    for cfg_path in args.model_configs:
        with open(cfg_path) as f:
            model_configs.append(json.load(f))

    # ----- Resolve prompts -----
    demo_cond = resolve_prompts(args.prompts_json, model_configs[0])

    # ----- Resolve CFG scales -----
    if args.cfg_scales is not None:
        cfg_scales = args.cfg_scales
    else:
        cfg_scales = (
            model_configs[0]
            .get("training", {})
            .get("demo", {})
            .get("demo_cfg_scales", [7.0])
        )
    cfg_scales = [float(s) for s in cfg_scales]
    print(f"CFG scales: {cfg_scales}")

    # ----- Prepare output directory and save prompt manifest -----
    os.makedirs(args.output_dir, exist_ok=True)
    prompts_out = os.path.join(args.output_dir, "prompts.json")
    with open(prompts_out, "w") as f:
        json.dump(demo_cond, f, indent=2)
    print(f"Prompts saved to {prompts_out}")

    # ----- Generate demos for each model -----
    for model_cfg, ckpt_path, model_name in zip(model_configs, ckpts, names):
        print(f"\n=== Model: {model_name} ===")
        if ckpt_path:
            print(f"  checkpoint: {ckpt_path}")

        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        model = load_model(model_cfg, ckpt_path, device)
        sample_size = model_cfg.get("sample_size", 524288)
        sample_rate = model_cfg.get("sample_rate", 44100)

        for prompt_idx, cond_dict in enumerate(demo_cond):
            prompt_preview = str(list(cond_dict.values())[0])[:60]
            print(f"  Prompt {prompt_idx:03d}: {prompt_preview!r}")

            for cfg_scale in cfg_scales:
                torch.manual_seed(args.seed)
                out_path = os.path.join(
                    out_dir, f"prompt_{prompt_idx:03d}_cfg{cfg_scale:.1f}.wav"
                )
                try:
                    with torch.inference_mode():
                        audio = generate_diffusion_cond(
                            model,
                            steps=args.steps,
                            cfg_scale=cfg_scale,
                            conditioning=[cond_dict],
                            batch_size=1,
                            sample_size=sample_size,
                            sample_rate=sample_rate,
                            seed=args.seed,
                            device=str(device),
                        )
                    # audio shape: (batch, channels, samples) → (channels, samples)
                    save_audio(audio[0], out_path, sample_rate)
                    print(f"    cfg={cfg_scale:.1f} → {out_path}")
                except Exception as exc:
                    print(f"    cfg={cfg_scale:.1f} FAILED: {type(exc).__name__}: {exc}")

        del model
        torch.cuda.empty_cache()

    print(f"\nDone. All demos saved under {args.output_dir}/")


if __name__ == "__main__":
    main()
