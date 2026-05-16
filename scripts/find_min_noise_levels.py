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
import threading
import time

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
    parser.add_argument("--t-steps", type=int, default=100,
                        help="Number of t values to sweep per sample (default: 100). Steps are "
                             "log-spaced via log(a/(a-t)) so coverage is coarse near t=0 and "
                             "fine near t=1 where the threshold crossing occurs.")
    parser.add_argument("--log-scale-stretch", type=float, default=1.00005,
                        help="Stretch parameter a > 1 for the t-grid spacing formula log(a/(a-t)). "
                             "Closer to 1 → more extreme concentration near t=1. Default: 1.00005.")

    parser.add_argument("--n-noise-samples", type=int, default=4,
                        help="Independent noise draws per t to average over (default: 4). "
                             "The sweep is vectorized: all t values are processed in one "
                             "forward pass per noise draw, so reducing this from 8→1 gives "
                             "an 8x speedup with minimal accuracy loss.")
    parser.add_argument("--tau", type=float, default=0.49, help="Probability threshold (default: 0.49 = 0.5 - epsilon).")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of files per forward pass (default: 1). The effective "
                             "GPU batch is batch_size × t_steps, so keep this small (1-4). "
                             "Increase if VRAM allows.")
    parser.add_argument("--resume", default=None, help="Path to an existing output JSON to resume from (skips already annotated files).")
    parser.add_argument("--reset", action="store_true", help="Ignore any existing output file and recompute everything from scratch.")
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
    # Resume: always try to load existing output first so the script is
    # safely re-runnable without --resume.  --resume is kept for back-compat
    # but is now a no-op (output path is always used).
    # ------------------------------------------------------------------
    results: dict = {}
    if not args.reset and os.path.isfile(args.output):
        with open(args.output) as f:
            results = json.load(f)
        filenames = [fn for fn in filenames if fn not in results]
        print(f"Resuming from {args.output}: {len(results)} already annotated, {len(filenames)} remaining.")
    elif not args.reset and args.resume and os.path.isfile(args.resume):
        with open(args.resume) as f:
            results = json.load(f)
        filenames = [fn for fn in filenames if fn not in results]
        print(f"Resuming from {args.resume}: {len(results)} already annotated, {len(filenames)} remaining.")

    # ------------------------------------------------------------------
    # t grid — log-spaced via x(t) = log(a/(a-t)), inverse t = a(1-exp(-x)).
    # Uniform spacing in x concentrates steps near t=1 (where a is the
    # singularity). Smaller a (closer to 1) → more extreme concentration.
    # ------------------------------------------------------------------
    a = args.log_scale_stretch
    t_end = min(1 - 1e-6, a - 1e-6)   # must stay below singularity at t=a
    x_end = float(np.log(a / (a - t_end)))
    x_grid = np.linspace(0, x_end, args.t_steps)
    t_grid = (a * (1 - np.exp(-x_grid))).tolist()

    # ------------------------------------------------------------------
    # Annotate in batches
    # ------------------------------------------------------------------
    load_timeout = 120  # seconds to wait for a single file load before skipping

    def _load_raw(path: str) -> torch.Tensor:
        ext = path.split(".")[-1]
        if ext == "pt":
            latents = torch.load(path, map_location="cpu")
        else:
            latents = torch.from_numpy(np.load(path).copy()).float()
        if pretransform_scale != 1.0:
            latents = latents / pretransform_scale
        return latents

    def load_latent(path: str) -> torch.Tensor:
        """Load with a thread-based timeout so a hung NFS read doesn't block forever.
        The background thread is daemon so it won't prevent process exit if it stays hung."""
        result: list = [None]
        exc: list = [None]

        def _worker():
            try:
                result[0] = _load_raw(path)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(load_timeout)
        if t.is_alive():
            raise TimeoutError(f"load timed out after {load_timeout}s (NFS likely unresponsive for this path)")
        if exc[0] is not None:
            raise exc[0]
        return result[0]

    batch_paths = []
    batch_tensors = []

    t_tensor = torch.tensor(t_grid, dtype=torch.float32, device=device)  # (T,)
    T = len(t_grid)

    def process_batch(paths, tensors):
        """Run the full t-sweep for a batch and write results."""
        nonlocal total_time, total_files
        t0 = time.time()
        B = len(tensors)
        L = max(t.shape[-1] for t in tensors)
        padded = torch.stack([
            F.pad(t, (0, L - t.shape[-1])) for t in tensors
        ]).to(device)  # (B, C, L)

        for path, per_t_probs in zip(paths, _sweep_batch_vectorized(padded)):
            t_min = find_t_min(per_t_probs, t_grid, args.tau)
            results[path] = t_min

        total_time += time.time() - t0
        total_files += B
        pbar.set_postfix({"s/file": f"{total_time / total_files:.2f}"})

    N = args.n_noise_samples

    @torch.no_grad()
    def _sweep_batch_vectorized(x: torch.Tensor):
        """
        Vectorized t-sweep: expand x across all T t-values and do N forward
        passes (one per noise draw) each of size T*B.

        Memory scales with T*B, so keep batch_size small (e.g. 1-4).
        With B=1 and T=100: effective GPU batch = 100. With N=4: 4 passes total
        vs the original T*N=400 passes.

        Old cost: T * N passes of B  →  New cost: N passes of T*B
        """
        B = x.shape[0]

        # Expand: (B, C, L) → (T*B, C, L)
        x_exp = x.unsqueeze(0).expand(T, -1, -1, -1).reshape(T * B, *x.shape[1:])

        # t values: (T,) → (T*B,)
        t_exp = t_tensor.unsqueeze(1).expand(T, B).reshape(T * B)
        alphas, sigmas = _get_alphas_sigmas(t_exp, diffusion_objective)  # (T*B, 1, 1)

        # N passes, each with independent noise
        probs_sum = torch.zeros(T * B, device=device)
        for _ in range(N):
            noise = torch.randn_like(x_exp)
            noised = x_exp * alphas + noise * sigmas
            logits = model(noised, t_exp)           # (T*B,)
            probs_sum += torch.sigmoid(logits)

        mean_probs = (probs_sum / N).reshape(T, B)  # (T, B)

        # Return as list[B] of list[T]
        return mean_probs.T.tolist()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    checkpoint_interval = 500  # write output every this many processed files
    processed_since_save = 0
    total_time = 0.0
    total_files = 0

    def save():
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    pbar = tqdm(filenames, desc="Annotating")
    for fn in pbar:
        try:
            latent = load_latent(fn)
        except (TimeoutError, RuntimeError, Exception) as e:
            tqdm.write(f"WARNING: skipping {fn}: {e}")
            continue

        batch_paths.append(fn)
        batch_tensors.append(latent)

        if len(batch_paths) >= args.batch_size:
            process_batch(batch_paths, batch_tensors)
            processed_since_save += len(batch_paths)
            batch_paths, batch_tensors = [], []

            if processed_since_save >= checkpoint_interval:
                save()
                processed_since_save = 0

    # Flush remaining partial batch
    if batch_paths:
        process_batch(batch_paths, batch_tensors)

    # Final save
    save()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    t_values = np.array(list(results.values()), dtype=float)
    print(f"\nDone. Annotated {len(t_values)} files → {args.output}")
    print(f"\n--- t_i^min summary (tau={args.tau}) ---")
    print(f"  mean:   {t_values.mean():.4f}")
    print(f"  std:    {t_values.std():.4f}")
    print(f"  min:    {t_values.min():.4f}")
    print(f"  p25:    {np.percentile(t_values, 25):.4f}")
    print(f"  median: {np.median(t_values):.4f}")
    print(f"  p75:    {np.percentile(t_values, 75):.4f}")
    print(f"  max:    {t_values.max():.4f}")
    # Histogram buckets
    buckets = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    print(f"\n  Distribution of t_i^min:")
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        count = int(((t_values >= lo) & (t_values < hi)).sum())
        bar = "#" * (count * 40 // max(len(t_values), 1))
        print(f"    [{lo:.2f}, {hi:.2f})  {count:6d}  {bar}")


if __name__ == "__main__":
    main()
