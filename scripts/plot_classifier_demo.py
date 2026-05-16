"""
plot_classifier_demo.py — Offline version of the DiTClassifier demo plot.

Loads a trained DiTClassifier from a model config + checkpoint, sweeps t from
0 → 1 over samples from both classes, and produces the merging-point plot that
the demo callback generates during training.

Files are collected via --dataset-config (recommended), which reuses the same
blacklist/whitelist filtering as training.  The label (0 or 1) is read from each
dataset entry's custom_metadata_module.

Usage:
    python scripts/plot_classifier_demo.py \
        --model-config stable_audio_tools/configs/model_configs/saosmall_ambient_classifier.json \
        --classifier-ckpt /deepfreeze/pnlong/ambient/classifier_runs/checkpoints/last.ckpt \
        --dataset-config stable_audio_tools/configs/dataset_configs/classifier_spdmx_jamendo.json \
        --output classifier_demo.pdf \
        --num-samples 8 \
        --t-steps 200 \
        --log-scale \
        --log-scale-stretch 1.05 \
        --device cuda
"""

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from stable_audio_tools.models.classifiers import create_dit_classifier_from_config, unstrict_load_checkpoint_weights
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from stable_audio_tools.data.dataset import get_latent_filenames


# ---------------------------------------------------------------------------
# File collection from dataset config
# ---------------------------------------------------------------------------

def _collect_files_from_dataset_config(dataset_config: dict, num_samples: int, seed: int):
    """
    Parse a pre_encoded dataset config, apply per-dataset blacklist/whitelist
    filtering (exactly as PreEncodedDataset does), and return two file lists:
        class0_files, class1_files
    The class label is determined by calling the custom_metadata_module for each
    dataset entry.
    """
    class_files = {0: [], 1: []}
    class_names = {0: "Low quality (label 0)", 1: "High quality (label 1)"}

    for entry in dataset_config.get("datasets", []):
        path_dir = entry["path"]
        latent_ext = entry.get("latent_extension", dataset_config.get("latent_extension", "npy"))
        whitelist_pth = entry.get("whitelist_pth", None)
        blacklist_pth = entry.get("blacklist_pth", None)

        files = get_latent_filenames(path_dir, [latent_ext])

        if whitelist_pth is not None:
            with open(whitelist_pth) as f:
                whitelist = set(x.split(".")[0] for x in f.read().splitlines())
            files = [fn for fn in files if os.path.splitext(os.path.basename(fn))[0] in whitelist]
            print(f"  [{entry['id']}] {len(files)} files after whitelist")
        elif blacklist_pth is not None:
            with open(blacklist_pth) as f:
                blacklist = set(Path(x).stem for x in f.read().splitlines())
            files = [fn for fn in files if Path(fn).stem not in blacklist]
            print(f"  [{entry['id']}] {len(files)} files after blacklist")
        else:
            print(f"  [{entry['id']}] {len(files)} files")

        # Determine class label via the custom metadata module
        label = None
        custom_module_path = entry.get("custom_metadata_module", None)
        if custom_module_path is not None:
            spec = importlib.util.spec_from_file_location("_meta_mod", custom_module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sample_meta = mod.get_custom_metadata({}, None)
            label = sample_meta.get("label", None)
            if label is not None:
                class_names[label] = entry["id"]

        if label is None:
            print(f"  [{entry['id']}] WARNING: could not determine label, skipping")
            continue

        random.seed(seed)
        random.shuffle(files)
        class_files[label].extend(files[:num_samples])

    return class_files[0], class_files[1], class_names[0], class_names[1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_file(path: str, device: torch.device) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return torch.from_numpy(arr).float().to(device)
    elif ext == ".pt":
        return torch.load(path, map_location=device).float()
    else:
        import torchaudio
        waveform, _ = torchaudio.load(path)
        return waveform.float().to(device)


# ---------------------------------------------------------------------------
# Noise / encode helpers (mirror DiTClassifierTrainingWrapper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _noise(x: torch.Tensor, t: torch.Tensor, diffusion_objective: str) -> torch.Tensor:
    if diffusion_objective in ("rectified_flow", "rf_denoiser"):
        alphas, sigmas = 1 - t, t
    elif diffusion_objective == "v":
        alphas, sigmas = get_alphas_sigmas(t)
    else:
        raise ValueError(f"Unknown diffusion_objective: {diffusion_objective}")
    return x * alphas[:, None, None] + torch.randn_like(x) * sigmas[:, None, None]


@torch.no_grad()
def _encode(x: torch.Tensor, pretransform, pre_encoded: bool) -> torch.Tensor:
    if pretransform is None:
        return x
    pretransform.to(x.device)
    if not pre_encoded:
        with torch.set_grad_enabled(pretransform.enable_grad):
            x = pretransform.encode(x)
    else:
        if hasattr(pretransform, "scale") and pretransform.scale != 1.0:
            x = x / pretransform.scale
    return x


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_plot(
    classifier,
    pretransform,
    pre_encoded: bool,
    diffusion_objective: str,
    class0_files: list,
    class1_files: list,
    t_steps: int,
    log_scale: bool,
    log_scale_stretch: float,
    class_names: list,
    device: torch.device,
    step_label: str,
    output_path: str,
) -> None:
    classifier.eval()

    if log_scale:
        # Log-spaced t-grid via x(t) = log(a/(a-t)), inverse t = a(1-exp(-x)).
        # Uses the same a as the x-axis display so the grid and plot are consistent.
        a = log_scale_stretch
        t_end = min(1 - 1e-4, a - 1e-6)
        x_end = float(np.log(a / (a - t_end)))
        t_grid = torch.tensor(
            a * (1 - np.exp(-np.linspace(0, x_end, t_steps))),
            dtype=torch.float32, device=device,
        )
    else:
        t_grid = torch.linspace(0, 1, t_steps, device=device)

    all_probs = {0: {"mean": [], "std": []}, 1: {"mean": [], "std": []}}

    for cls_idx, files in enumerate([class0_files, class1_files]):
        label = class_names[cls_idx]
        print(f"  [{label}] pre-loading {len(files)} files ...")

        clean_latents = []
        for path in files:
            x = _load_file(path, device)
            if x.ndim == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.ndim == 2:
                x = x.unsqueeze(0)
            x = _encode(x, pretransform, pre_encoded)
            if x.isnan().any():
                print(f"  [{label}] skipping NaN file: {os.path.basename(path)}")
                continue
            clean_latents.append(x)
        print(f"  [{label}] {len(clean_latents)}/{len(files)} files are clean")

        if not clean_latents:
            print(f"  [{label}] WARNING: no clean files, class will be missing from plot")

        for t_val in t_grid:
            probs_at_t = []
            for x in clean_latents:
                t = torch.full((x.shape[0],), t_val.item(), device=device)
                noised = _noise(x, t, diffusion_objective)
                logits = classifier(noised, t)
                probs_at_t.append(torch.sigmoid(logits).mean().item())
            if probs_at_t:
                arr = np.array(probs_at_t)
                all_probs[cls_idx]["mean"].append(float(arr.mean()))
                all_probs[cls_idx]["std"].append(float(arr.std()))
            else:
                all_probs[cls_idx]["mean"].append(float("nan"))
                all_probs[cls_idx]["std"].append(float("nan"))

        mean_p = float(np.nanmean(all_probs[cls_idx]["mean"])) if all_probs[cls_idx]["mean"] else float("nan")
        print(f"  [{label}] mean p(high quality) across all t: {mean_p:.3f}")

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 4))

    t_np = t_grid.cpu().numpy()
    if log_scale:
        a = log_scale_stretch
        x_vals = np.log(a / (a - t_np))
    else:
        x_vals = t_np

    mean0 = np.array(all_probs[0]["mean"])
    mean1 = np.array(all_probs[1]["mean"])
    std0  = np.array(all_probs[0]["std"])
    std1  = np.array(all_probs[1]["std"])

    # ------------------------------------------------------------------
    # Compute marker locations (interpolated)
    # A) first t where high-quality (class 1) curve drops below 0.5
    # B) where high-quality and low-quality curves cross
    # ------------------------------------------------------------------
    marker_style = dict(color="gray", linestyle="--", linewidth=1, alpha=0.7)

    t_below50 = x_below50 = None
    for i in range(len(mean1)):
        if not np.isnan(mean1[i]) and mean1[i] < 0.5:
            if i == 0:
                t_below50, x_below50 = t_np[0], x_vals[0]
            else:
                frac = (0.5 - mean1[i-1]) / (mean1[i] - mean1[i-1])
                t_below50 = float(t_np[i-1] + frac * (t_np[i] - t_np[i-1]))
                x_below50 = float(x_vals[i-1] + frac * (x_vals[i] - x_vals[i-1]))
            break

    t_cross = x_cross = y_cross = None
    diff = mean1 - mean0
    for i in range(1, len(diff)):
        if not (np.isnan(diff[i-1]) or np.isnan(diff[i])) and diff[i-1] > 0 >= diff[i]:
            frac = diff[i-1] / (diff[i-1] - diff[i])
            t_cross = float(t_np[i-1] + frac * (t_np[i] - t_np[i-1]))
            x_cross = float(x_vals[i-1] + frac * (x_vals[i] - x_vals[i-1]))
            y_cross = float(mean1[i-1] + frac * (mean1[i] - mean1[i-1]))
            break

    print("--- Marker values ---")
    print(f"  High quality first drops below 0.5:  t = {t_below50:.6f}" if t_below50 is not None else "  High quality never drops below 0.5")
    print(f"  High quality / low quality crossing:  t = {t_cross:.6f},  p = {y_cross:.4f}" if t_cross is not None else "  Curves do not cross")

    colors = ["#e05c5c", "#5c8fe0"]
    for cls_idx, (mean, std) in enumerate([(mean0, std0), (mean1, std1)]):
        ax.plot(x_vals, mean, color=colors[cls_idx], linewidth=2, label=class_names[cls_idx])
        ax.fill_between(
            x_vals,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=colors[cls_idx],
            alpha=0.2,
        )

    # Decision boundary — no legend entry
    ax.axhline(0.5, **marker_style)

    # Vertical + horizontal markers
    if x_below50 is not None:
        ax.axvline(x_below50, **marker_style)
    if x_cross is not None:
        ax.axvline(x_cross, **marker_style)
        ax.axhline(y_cross, **marker_style)

    ax.set_xlabel("Noise level (t)")
    ax.set_ylabel("p(high quality | noisy x, t)")
    ax.set_title(f"Classifier probability vs noise level — {step_label}")
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(0, 1)

    if log_scale:
        tick_x = np.linspace(x_vals[0], x_vals[-1], 9)
        tick_t = a * (1 - np.exp(-tick_x))
        ax.set_xticks(tick_x)
        ax.set_xticklabels([f"{v:.4f}" for v in tick_t])

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate the DiTClassifier merging-point plot from a saved checkpoint."
    )
    parser.add_argument("--model-config", required=True,
                        help="Path to the model JSON config (e.g. saosmall_ambient_classifier.json).")
    parser.add_argument("--classifier-ckpt", required=True,
                        help="Path to classifier checkpoint (.ckpt or .safetensors).")
    parser.add_argument("--dataset-config", required=True,
                        help="Path to the dataset JSON config used for training (e.g. classifier_spdmx_jamendo.json). "
                             "Blacklists/whitelists and labels are read from here.")
    parser.add_argument("--output", default="classifier_demo.pdf",
                        help="Output path (format inferred from extension: .pdf, .png, .svg, ...).")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of files per class to sweep over.")
    parser.add_argument("--t-steps", type=int, default=200,
                        help="Number of t values to sweep (default: 200). Steps are log-spaced "
                             "via log(a/(a-t)) so coverage is coarse near t=0 and fine near t=1 "
                             "where distributions merge and the threshold crossing occurs.")
    parser.add_argument("--log-scale", action="store_true",
                        help="Use log(a/(a-t)) x-axis scaling.")
    parser.add_argument("--log-scale-stretch", type=float, default=1.00005,
                        help="Stretch parameter a > 1 shared by both the t-grid sampling and "
                             "the x-axis display formula log(a/(a-t)). Closer to 1 → more "
                             "extreme concentration near t=1. Default: 1.00005.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)

    # ---- Load configs -------------------------------------------------------
    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    # ---- Build classifier ---------------------------------------------------
    print(f"Building model from {args.model_config} ...")
    classifier = create_dit_classifier_from_config(model_config)
    print(f"Loading weights from {args.classifier_ckpt} ...")
    unstrict_load_checkpoint_weights(classifier, args.classifier_ckpt)
    classifier = classifier.to(device).eval()

    model_cfg = model_config.get("model", {})
    training_cfg = model_config.get("training", {})
    diffusion_objective = model_cfg.get("diffusion", {}).get("diffusion_objective", "rectified_flow")
    pre_encoded = training_cfg.get("pre_encoded", True)

    pretransform = None
    if "pretransform" in model_cfg:
        from stable_audio_tools.models.factory import create_pretransform_from_config
        sample_rate = model_config.get("sample_rate", 44100)
        pretransform = create_pretransform_from_config(model_cfg["pretransform"], sample_rate)
        pretransform = pretransform.to(device).eval()

    # ---- Collect files via dataset config -----------------------------------
    print("Collecting files from dataset config ...")
    class0_files, class1_files, class0_name, class1_name = _collect_files_from_dataset_config(
        dataset_config, args.num_samples, args.seed
    )
    if not class0_files:
        raise RuntimeError("No class-0 files found — check dataset config and blacklist.")
    if not class1_files:
        raise RuntimeError("No class-1 files found — check dataset config and whitelist.")

    # ---- Generate plot ------------------------------------------------------
    ckpt_name = os.path.basename(args.classifier_ckpt)
    print(f"Sweeping {args.t_steps} t-values ...")
    generate_plot(
        classifier=classifier,
        pretransform=pretransform,
        pre_encoded=pre_encoded,
        diffusion_objective=diffusion_objective,
        class0_files=class0_files,
        class1_files=class1_files,
        t_steps=args.t_steps,
        log_scale=args.log_scale,
        log_scale_stretch=args.log_scale_stretch,
        class_names=[class0_name, class1_name],
        device=device,
        step_label=ckpt_name,
        output_path=args.output,
    )

    print(f"Saved plot → {args.output}")


if __name__ == "__main__":
    main()
