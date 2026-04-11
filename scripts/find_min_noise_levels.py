"""
find_min_noise_levels.py — Per-sample min_noise_level annotation using a trained DiTClassifier.

Implements the sample-dependent annotation scheme from eq. 3.2 / sample-dependent
variant of the Ambient Diffusion paper:

    t_i^min = inf { t in [0, T] : E[c_theta(w_t | w_0^(i))] > tau }

For each pre-encoded latent file in --data-dir, sweeps t from 0 → 1, averages the
classifier probability over --n-noise-samples independent noise draws, and finds the
lowest t at which the mean probability exceeds --tau.

Results are written to --output as a JSON mapping absolute file path → t_i^min.

Usage:
    python scripts/find_min_noise_levels.py \
        --model-config stable_audio_tools/configs/model_configs/saosmall_ambient_classifier.json \
        --classifier-ckpt /deepfreeze/pnlong/ambient/classifier_runs/checkpoints/last.ckpt \
        --data-dir /deepfreeze/share/sPDMX_stems_extracted \
        --output /deepfreeze/pnlong/ambient/spdmx_min_noise_levels.json \
        --latent-extension npy \
        --t-steps 100 \
        --n-noise-samples 8 \
        --tau 0.49 \
        --device cuda \
        --batch-size 32

The output JSON can be consumed by a custom metadata module (see
stable_audio_tools/configs/dataset_configs/custom_metadata/min_noise_level_lookup.py).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.classifiers import unstrict_load_checkpoint_weights
from stable_audio_tools.data.dataset import get_latent_filenames
from stable_audio_tools.inference.sampling import get_alphas_sigmas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_alphas_sigmas(t: torch.Tensor, diffusion_objective: str):
    """Return (alphas, sigmas) shaped (B, 1, 1) for a given diffusion objective."""
    if diffusion_objective in ("rectified_flow", "rf_denoiser"):
        alphas = 1 - t
        sigmas = t
    elif diffusion_objective == "v":
        alphas, sigmas = get_alphas_sigmas(t)
    else:
        raise ValueError(f"Unknown diffusion_objective: {diffusion_objective}")
    return alphas[:, None, None], sigmas[:, None, None]


@torch.no_grad()
def classify_at_t(model, x: torch.Tensor, t_val: float, n_noise_samples: int,
                  diffusion_objective: str) -> float:
    """
    Returns mean p(label=1 | x_t, t) averaged over n_noise_samples noise draws.
    x: (B, C, L) latents on the model's device.
    """
    B = x.shape[0]
    t = torch.full((B,), t_val, device=x.device, dtype=x.dtype)
    alphas, sigmas = _get_alphas_sigmas(t, diffusion_objective)

    probs_sum = torch.zeros(B, device=x.device)
    for _ in range(n_noise_samples):
        noise = torch.randn_like(x)
        noised = x * alphas + noise * sigmas
        logits = model(noised, t)            # (B,)
        probs_sum += torch.sigmoid(logits)   # (B,)

    return (probs_sum / n_noise_samples).cpu()  # (B,)


def find_t_min(mean_probs_per_t: list, t_grid: list, tau: float) -> float:
    """
    Find the infimum t in t_grid where mean_probs_per_t[i] > tau.
    Returns 1.0 (max noise) if the threshold is never crossed.
    """
    for t_val, p in zip(t_grid, mean_probs_per_t):
        if p > tau:
            return float(t_val)
    return float(t_grid[-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Annotate pre-encoded latents with per-sample min_noise_level.")
    parser.add_argument("--model-config", required=True, help="Path to the dit_classifier model config JSON.")
    parser.add_argument("--classifier-ckpt", required=True, help="Path to trained classifier checkpoint (.ckpt or .safetensors).")
    parser.add_argument("--data-dir", required=True, help="Root directory of pre-encoded .npy latents to annotate.")
    parser.add_argument("--output", required=True, help="Output JSON file path (filepath → t_i_min).")
    parser.add_argument("--latent-extension", default="npy", help="Latent file extension (default: npy).")
    parser.add_argument("--blacklist-pth", default=None, help="Optional path to a blacklist file (one stem per line).")
    parser.add_argument("--t-steps", type=int, default=100, help="Number of evenly-spaced t values to sweep (default: 100).")
    parser.add_argument("--n-noise-samples", type=int, default=8, help="Noise draws per t to average over (default: 8).")
    parser.add_argument("--tau", type=float, default=0.49, help="Probability threshold (default: 0.49 = 0.5 - epsilon).")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for forward passes (default: 32).")
    parser.add_argument("--resume", default=None, help="Path to an existing output JSON to resume from (skips already annotated files).")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load model config
    # ------------------------------------------------------------------
    with open(args.model_config) as f:
        model_config = json.load(f)

    diffusion_objective = (
        model_config["model"]
        .get("diffusion", {})
        .get("diffusion_objective", "rectified_flow")
    )

    # Pretransform scale (applied to pre-encoded latents, same as DiTClassifierTrainingWrapper._encode)
    pretransform_scale = (
        model_config["model"]
        .get("pretransform", {})
        .get("scale", 1.0)
    )

    # ------------------------------------------------------------------
    # Build model and load trained classifier weights
    # ------------------------------------------------------------------
    print("Building classifier model...")
    model = create_model_from_config(model_config)  # DiTClassifier; loads DiT backbone from checkpoint_path in config

    print(f"Loading trained classifier weights from {args.classifier_ckpt}...")
    loaded, skipped = unstrict_load_checkpoint_weights(model, args.classifier_ckpt)
    print(f"  Loaded {len(loaded)} keys, skipped {len(skipped)} keys.")

    model = model.to(device).eval()

    # ------------------------------------------------------------------
    # Collect latent files
    # ------------------------------------------------------------------
    print(f"Scanning {args.data_dir} for .{args.latent_extension} files...")
    filenames = get_latent_filenames(args.data_dir, [args.latent_extension])

    if args.blacklist_pth is not None:
        from pathlib import Path
        with open(args.blacklist_pth) as f:
            blacklist = set(Path(x.strip()).stem for x in f if x.strip())
        filenames = [fn for fn in filenames if Path(fn).stem not in blacklist]

    print(f"Found {len(filenames)} files to annotate.")

    # ------------------------------------------------------------------
    # Resume from existing output if requested
    # ------------------------------------------------------------------
    results: dict = {}
    if args.resume and os.path.isfile(args.resume):
        with open(args.resume) as f:
            results = json.load(f)
        filenames = [fn for fn in filenames if fn not in results]
        print(f"Resuming: {len(results)} already annotated, {len(filenames)} remaining.")

    # ------------------------------------------------------------------
    # t grid
    # ------------------------------------------------------------------
    t_grid = torch.linspace(0, 1, args.t_steps).tolist()

    # ------------------------------------------------------------------
    # Annotate in batches
    # ------------------------------------------------------------------
    def load_latent(path: str) -> torch.Tensor:
        ext = path.split(".")[-1]
        if ext == "pt":
            latents = torch.load(path, map_location="cpu")
        else:
            latents = torch.from_numpy(np.load(path))
        if pretransform_scale != 1.0:
            latents = latents / pretransform_scale
        return latents  # (C, L)

    batch_paths = []
    batch_tensors = []
    max_len = 0

    def process_batch(paths, tensors):
        """Run the full t-sweep for a batch and write results."""
        # Pad all tensors in the batch to the same length
        L = max(t.shape[-1] for t in tensors)
        padded = torch.stack([
            F.pad(t, (0, L - t.shape[-1])) for t in tensors
        ]).to(device)  # (B, C, L)

        for path, per_t_probs in zip(paths, _sweep_batch(padded)):
            t_min = find_t_min(per_t_probs, t_grid, args.tau)
            results[path] = t_min

    def _sweep_batch(x: torch.Tensor):
        """Yields per-sample list-of-floats (one per t value) for each item in the batch."""
        B = x.shape[0]
        per_sample_probs = [[] for _ in range(B)]
        for t_val in t_grid:
            probs = classify_at_t(model, x, t_val, args.n_noise_samples, diffusion_objective)
            for i, p in enumerate(probs.tolist()):
                per_sample_probs[i].append(p)
        return per_sample_probs

    checkpoint_interval = 500  # write output every this many files
    processed_since_save = 0

    for fn in tqdm(filenames, desc="Annotating"):
        try:
            latent = load_latent(fn)
        except Exception as e:
            print(f"WARNING: could not load {fn}: {e}")
            continue

        batch_paths.append(fn)
        batch_tensors.append(latent)

        if len(batch_paths) >= args.batch_size:
            process_batch(batch_paths, batch_tensors)
            processed_since_save += len(batch_paths)
            batch_paths, batch_tensors = [], []

            if processed_since_save >= checkpoint_interval:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                processed_since_save = 0

    # Flush remaining
    if batch_paths:
        process_batch(batch_paths, batch_tensors)

    # Final save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Annotated {len(results)} files → {args.output}")


if __name__ == "__main__":
    main()
