"""
eval.py — Generate audio from a trained diffusion model and compute evaluation metrics.

Two metrics are computed:
  FAD  (Fréchet Audio Distance) — distribution-level quality vs. a reference corpus.
       Uses fadtk with one or more embedding backends (VGGish, CLAP, EnCodec, …).
  CLAP Score — per-sample text-audio cosine similarity using laion-clap.
       Measures how well each generated clip matches its conditioning prompt.

Requires:
  pip install fadtk          # new dependency
  laion-clap already in setup.py

Usage — generate from checkpoint, then evaluate both metrics:
    python scripts/eval.py \\
        --model-config stable_audio_tools/configs/model_configs/saosmall_ambient_loss.json \\
        --ckpt /deepfreeze/pnlong/ambient/runs/checkpoints/last.ckpt \\
        --reference-dir /path/to/reference/audio \\
        --output-dir /tmp/eval_results \\
        --num-samples 50

Usage — multiple FAD embeddings:
    python scripts/eval.py ... --embedding vggish --embedding clap-laion-music

Usage — skip generation (audio already in <output-dir>/generated/):
    python scripts/eval.py \\
        --reference-dir /path/to/reference/audio \\
        --output-dir /tmp/eval_results \\
        --skip-generation

Usage — CLAP Score only (no reference corpus needed):
    python scripts/eval.py ... --no-fad

Usage — GPU (pick the card with CUDA_VISIBLE_DEVICES; default device is CPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/eval.py ... --gpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure the repo root is on the path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Ensure scripts/ itself is on the path for _utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_audio_tools.inference.generation import generate_diffusion_cond
from _utils import load_model, resolve_prompts, save_audio


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _prompt_postfix(text: str, max_len: int = 55) -> str:
    t = " ".join((text or "").split())
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t if t else "(no prompt)"


def generate_samples(
    model_config: dict,
    ckpt_path: str,
    prompts: list[dict],
    generated_dir: Path,
    num_samples: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    device: torch.device,
) -> list[tuple[Path, str]]:
    """Generate num_samples audio clips, cycling through prompts.

    Each clip is saved as <generated_dir>/<idx:05d>.wav alongside a sidecar
    <idx:05d>.txt containing the prompt text (used later for CLAP Score).

    Returns a list of (wav_path, prompt_text) pairs for all generated clips.
    """
    sample_size = model_config.get("sample_size", 524288)
    sample_rate = model_config.get("sample_rate", 44100)

    model = load_model(model_config, ckpt_path, device)

    pairs = []
    with tqdm(
        total=num_samples,
        desc="Generate",
        unit="clip",
        dynamic_ncols=True,
    ) as pbar:
        for i in range(num_samples):
            cond_dict = prompts[i % len(prompts)]
            prompt_text = cond_dict.get("prompt", "")
            wav_path = generated_dir / f"{i:05d}.wav"
            txt_path = generated_dir / f"{i:05d}.txt"

            pbar.set_postfix_str(_prompt_postfix(prompt_text), refresh=False)

            torch.manual_seed(seed + i)
            try:
                with torch.inference_mode():
                    audio = generate_diffusion_cond(
                        model,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        conditioning=[cond_dict],
                        batch_size=1,
                        sample_size=sample_size,
                        sample_rate=sample_rate,
                        seed=seed + i,
                        device=str(device),
                        disable_tqdm=True,
                        quiet=True,
                    )
                save_audio(audio[0], str(wav_path), sample_rate)
                txt_path.write_text(prompt_text)
                pairs.append((wav_path, prompt_text))
            except Exception as exc:
                pbar.set_postfix_str(f"FAILED {type(exc).__name__}", refresh=True)
                tqdm.write(f"  [{i + 1}/{num_samples}] FAILED: {type(exc).__name__}: {exc}")
            pbar.update(1)

    del model
    torch.cuda.empty_cache()
    return pairs


def load_existing_pairs(generated_dir: Path) -> list[tuple[Path, str | None]]:
    """Collect (wav_path, prompt_text | None) pairs from an existing directory.

    Prompt text is read from a sidecar .txt file if present, otherwise None
    (CLAP Score will be skipped for files without a sidecar).
    """
    wav_paths = sorted(generated_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(
            f"No .wav files found in {generated_dir}. "
            "Run without --skip-generation to generate audio first."
        )
    pairs = []
    for wav_path in wav_paths:
        txt_path = wav_path.with_suffix(".txt")
        prompt_text = txt_path.read_text().strip() if txt_path.exists() else None
        pairs.append((wav_path, prompt_text))
    return pairs


# ---------------------------------------------------------------------------
# FAD via fadtk
# ---------------------------------------------------------------------------

def compute_fad(
    reference_dir: str,
    generated_dir: Path,
    embeddings: list[str],
) -> dict[str, float]:
    """Compute FAD for each requested embedding model. Returns {emb_name: score}."""
    try:
        from fadtk import FrechetAudioDistance
        from fadtk.model_loader import ModelLoader
    except ImportError as exc:
        raise ImportError(
            "fadtk is required for FAD computation. Install it with: pip install fadtk"
        ) from exc

    scores = {}
    for emb_name in embeddings:
        print(f"\n  FAD [{emb_name}]: computing embeddings …")
        try:
            fad = FrechetAudioDistance(ModelLoader(emb_name), audio_load_worker=4)
            score = fad.score(reference_dir, str(generated_dir))
            scores[emb_name] = float(score)
            print(f"  FAD [{emb_name}]: {score:.4f}")
        except Exception as exc:
            print(f"  FAD [{emb_name}]: FAILED — {type(exc).__name__}: {exc}")
            scores[emb_name] = None
    return scores


# ---------------------------------------------------------------------------
# CLAP Score via laion-clap
# ---------------------------------------------------------------------------

def compute_clap_score(
    pairs: list[tuple[Path, str | None]],
    clap_model_path: str | None,
    device: torch.device,
) -> float | None:
    """Compute mean cosine similarity between text prompts and generated audio.

    Returns None if no pairs have sidecar prompt text.
    """
    try:
        import laion_clap
    except ImportError as exc:
        raise ImportError(
            "laion-clap is required for CLAP Score. It should already be installed "
            "(listed in setup.py). Run: pip install laion-clap"
        ) from exc

    valid_pairs = [(p, t) for p, t in pairs if t is not None]
    if not valid_pairs:
        print(
            "  CLAP Score: skipped — no sidecar .txt prompt files found in generated_dir. "
            "Re-run generation (without --skip-generation) to create them."
        )
        return None

    skipped = len(pairs) - len(valid_pairs)
    if skipped:
        print(f"  CLAP Score: {skipped} file(s) missing sidecar .txt — excluded from scoring.")

    print(f"\n  CLAP Score: loading model …")
    clap = laion_clap.CLAP_Module(enable_fusion=True, device=str(device))
    # load_ckpt with no args downloads the default checkpoint;
    # a local path can be passed to use a custom checkpoint.
    if clap_model_path:
        clap.load_ckpt(clap_model_path)
    else:
        clap.load_ckpt()

    wav_paths = [str(p) for p, _ in valid_pairs]
    texts = [t for _, t in valid_pairs]

    print(f"  CLAP Score: scoring {len(valid_pairs)} clips …")
    with torch.no_grad():
        audio_embs = clap.get_audio_embedding_from_filelist(wav_paths, use_tensor=True)
        text_embs = clap.get_text_embedding(texts, use_tensor=True)

    # Both tensors: (N, D) — compute per-sample cosine similarity then average
    sims = F.cosine_similarity(audio_embs, text_embs, dim=-1)
    score = float(sims.mean().item())
    print(f"  CLAP Score: {score:.4f}  (mean over {len(valid_pairs)} clips)")
    return score


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    """Print a human-readable summary table to stdout."""
    width = 40
    print("\n" + "=" * width)
    print("  EVALUATION RESULTS")
    print("=" * width)
    print(f"  Generated dir : {results['generated_dir']}")
    print(f"  Reference dir : {results['reference_dir']}")
    print(f"  # generated   : {results['num_generated']}")
    print("-" * width)
    fad = results.get("fad", {})
    if fad:
        for emb, val in fad.items():
            val_str = f"{val:.4f}" if val is not None else "FAILED"
            print(f"  FAD [{emb:20s}]: {val_str}")
    else:
        print("  FAD           : not computed")
    cs = results.get("clap_score")
    if cs is not None:
        print(f"  CLAP Score    : {cs:.4f}")
    else:
        print("  CLAP Score    : not computed")
    print("=" * width)


def save_results(results: dict, output_dir: Path) -> None:
    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate audio from a diffusion checkpoint and compute evaluation metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    gen_group = parser.add_argument_group("Model / generation (required unless --skip-generation)")
    gen_group.add_argument(
        "--model-config", dest="model_config", metavar="PATH",
        help="Path to the model JSON config.",
    )
    gen_group.add_argument(
        "--ckpt", dest="ckpt", default="", metavar="PATH",
        help="Path to a .ckpt or .safetensors checkpoint.",
    )
    gen_group.add_argument(
        "--prompts-json", default=None, metavar="PATH",
        help="JSON file with conditioning dicts. Falls back to training.demo.demo_cond "
             "in the model config, then built-in defaults.",
    )
    gen_group.add_argument(
        "--num-samples", type=int, default=50, metavar="N",
        help="Total audio clips to generate (default: 50; FAD is unreliable below ~50).",
    )
    gen_group.add_argument(
        "--steps", type=int, default=100,
        help="Diffusion steps (default: 100).",
    )
    gen_group.add_argument(
        "--cfg-scale", type=float, default=7.0,
        help="Classifier-free guidance scale (default: 7.0).",
    )
    gen_group.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0); seed+i is used for clip i.",
    )
    gen_group.add_argument(
        "--skip-generation", action="store_true",
        help="Skip generation. Expects WAV files to already exist in "
             "<output-dir>/generated/. CLAP Score also requires sidecar .txt files.",
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--reference-dir", metavar="DIR",
        help="Directory of reference (real) WAV files. Required for FAD.",
    )
    eval_group.add_argument(
        "--embedding", dest="embeddings", action="append",
        default=None, metavar="NAME",
        help="FAD embedding backend (default: vggish). Repeat to run multiple, "
             "e.g. --embedding vggish --embedding clap-laion-music. "
             "See fadtk docs for all supported names.",
    )
    eval_group.add_argument(
        "--clap-model", default=None, metavar="PATH_OR_NAME",
        help="Path to a LAION-CLAP checkpoint for CLAP Score. "
             "Omit to use the default downloaded checkpoint.",
    )
    eval_group.add_argument(
        "--no-fad", action="store_true",
        help="Skip FAD computation.",
    )
    eval_group.add_argument(
        "--no-clap-score", action="store_true",
        help="Skip CLAP Score computation.",
    )

    parser.add_argument(
        "--output-dir", default="eval_results", metavar="DIR",
        help="Root output directory (default: eval_results). "
             "Generated audio goes to <output-dir>/generated/.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CUDA for inference and CLAP (omit for CPU). "
             "Set CUDA_VISIBLE_DEVICES to select the GPU.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ----- Validate -----
    if not args.skip_generation and not args.model_config:
        parser.error("--model-config is required unless --skip-generation is set.")
    if not args.no_fad and not args.reference_dir:
        parser.error("--reference-dir is required for FAD (use --no-fad to skip).")

    if args.embeddings is None:
        args.embeddings = ["vggish"]

    if args.gpu:
        if not torch.cuda.is_available():
            parser.error("--gpu requested but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # ----- Phase 1: Generation -----
    if args.skip_generation:
        print(f"Skipping generation — loading existing audio from {generated_dir}")
        pairs = load_existing_pairs(generated_dir)
        print(f"Found {len(pairs)} WAV files.")
    else:
        with open(args.model_config) as f:
            model_config = json.load(f)
        prompts = resolve_prompts(args.prompts_json, model_config)

        # Save the prompt manifest so results are reproducible
        manifest_path = output_dir / "prompts.json"
        with open(manifest_path, "w") as f:
            json.dump(prompts, f, indent=2)
        print(f"Prompts manifest saved to {manifest_path}")

        print(f"\n=== Phase 1: Generating {args.num_samples} clips ===")
        pairs = generate_samples(
            model_config=model_config,
            ckpt_path=args.ckpt,
            prompts=prompts,
            generated_dir=generated_dir,
            num_samples=args.num_samples,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            device=device,
        )
        print(f"Generated {len(pairs)} clips → {generated_dir}")

    results: dict = {
        "num_generated": len(pairs),
        "reference_dir": args.reference_dir,
        "generated_dir": str(generated_dir),
        "fad": {},
        "clap_score": None,
    }

    # ----- Phase 2: FAD -----
    if not args.no_fad:
        print(f"\n=== Phase 2: FAD (embeddings: {args.embeddings}) ===")
        results["fad"] = compute_fad(
            reference_dir=args.reference_dir,
            generated_dir=generated_dir,
            embeddings=args.embeddings,
        )

    # ----- Phase 3: CLAP Score -----
    if not args.no_clap_score:
        print("\n=== Phase 3: CLAP Score ===")
        results["clap_score"] = compute_clap_score(
            pairs=pairs,
            clap_model_path=args.clap_model,
            device=device,
        )

    # ----- Phase 4: Output -----
    print_summary(results)
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
