"""
Batch-run scripts/eval.py on Song Describer prompts and aggregate metrics.

Use the project env:
  mamba activate stable_audio_tools

If activate fails in a non-interactive shell:
  micromamba run -n stable_audio_tools python ...

Typical flow:
  1. python scripts/song_describer_eval/preprocess.py
       (uses <sdd-root>/audio by default when present)
  2. python scripts/song_describer_eval/run_eval.py --runs-json runs.json
  3. Inspect /deepfreeze/pnlong/ambient/song_describer_eval/summary.csv and summary.png

Any extra CLI tokens are forwarded to eval.py (e.g. --embedding clap-laion-music).
Use --gpu to add eval.py's --gpu flag for each run (CUDA_VISIBLE_DEVICES selects the GPU).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval.py"

DEFAULT_STORAGE = Path("/deepfreeze/pnlong/ambient/song_describer_eval")
DEFAULT_CACHE = DEFAULT_STORAGE / "cache"


def load_runs(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"{path} must contain a 'runs' list")
    return runs


def summarize_runs(storage_root: Path) -> None:
    runs_root = storage_root / "runs"
    rows: list[dict] = []
    if not runs_root.is_dir():
        print(f"No runs directory: {runs_root}", file=sys.stderr)
        return

    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        metrics_path = d / "metrics.json"
        if not metrics_path.is_file():
            continue
        with metrics_path.open(encoding="utf-8") as f:
            m = json.load(f)
        row: dict = {"run": d.name}
        cs = m.get("clap_score")
        row["clap_score"] = cs if cs is not None else ""
        fad = m.get("fad") or {}
        for emb, val in fad.items():
            key = f"fad_{emb}"
            if val is None:
                row[key] = ""
            else:
                row[key] = float(val)
        rows.append(row)

    if not rows:
        print("No metrics.json files found under runs/.", file=sys.stderr)
        return

    # CSV summary
    fad_keys = sorted({k for r in rows for k in r if k.startswith("fad_")})
    all_keys = ["run", "clap_score"] + fad_keys

    csv_path = storage_root / "summary.csv"
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_keys})
    print(f"Wrote {csv_path}")

    # Plot(s)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped summary.png (CSV only).", file=sys.stderr)
        return

    names = [r["run"] for r in rows]
    n = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, n * 0.45), 5.0))
    ax_clap, ax_fad = axes[0], axes[1]

    clap_vals = [float(r["clap_score"]) if r.get("clap_score") != "" else float("nan") for r in rows]
    x = range(n)
    ax_clap.bar(x, clap_vals, color="steelblue")
    ax_clap.set_xticks(list(x))
    ax_clap.set_xticklabels(names, rotation=45, ha="right")
    ax_clap.set_ylabel("CLAP score (↑ better)")
    ax_clap.set_title("Song Describer — text–audio alignment")

    fad_keys = [k for k in all_keys if k.startswith("fad_")]
    if fad_keys:
        width = 0.8 / len(fad_keys)
        for i, fk in enumerate(fad_keys):
            offs = [xi + (i - len(fad_keys) / 2 + 0.5) * width for xi in x]
            vals = [float(r[fk]) if r.get(fk) not in ("", None) else float("nan") for r in rows]
            label = fk.replace("fad_", "")
            ax_fad.bar(offs, vals, width=width, label=label)
        ax_fad.set_xticks(list(x))
        ax_fad.set_xticklabels(names, rotation=45, ha="right")
        ax_fad.set_ylabel("FAD (↓ better)")
        ax_fad.set_title("Song Describer — distribution vs reference")
        ax_fad.legend(fontsize=8)
    else:
        ax_fad.text(0.5, 0.5, "No FAD columns", ha="center", va="center")
        ax_fad.set_axis_off()

    fig.tight_layout()
    png_path = storage_root / "summary.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_STORAGE,
        help="Top-level storage (runs go to <this>/runs/<name>/)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Directory with prompts.json and reference_wavs/ from preprocess.py",
    )
    parser.add_argument(
        "--runs-json",
        type=Path,
        default=None,
        help='JSON file: {"runs":[{"name","model_config","ckpt"}, ...]}',
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Single run: logical name (output under runs/<name>/)",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Single run: model JSON path",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Single run: checkpoint path",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only rebuild summary.csv / summary.png from existing metrics",
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip aggregate table/plot after batch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print eval.py commands without executing",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Add --gpu to each eval.py subprocess (use CUDA_VISIBLE_DEVICES to pick the GPU)",
    )

    args, eval_extra = parser.parse_known_args()

    if args.summarize_only:
        summarize_runs(args.storage_root.resolve())
        return

    runs: list[dict] = load_runs(args.runs_json)
    if args.run_name and args.model_config and args.ckpt:
        runs.append(
            {
                "name": args.run_name,
                "model_config": str(args.model_config.resolve()),
                "ckpt": str(args.ckpt.resolve()),
            }
        )
    if not runs:
        parser.error("Provide --runs-json and/or --run-name with --model-config and --ckpt")

    prompts_json = args.cache_dir / "prompts.json"
    ref_dir = args.cache_dir / "reference_wavs"
    if not prompts_json.is_file():
        raise FileNotFoundError(f"Missing {prompts_json}. Run preprocess.py first.")

    try:
        with prompts_json.open(encoding="utf-8") as f:
            prompt_list = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid prompts.json: {e}") from e

    num_samples = len(prompt_list)
    if num_samples == 0:
        raise SystemExit("prompts.json is empty")

    use_fad = ref_dir.is_dir() and any(ref_dir.iterdir())
    if not use_fad:
        print(
            f"Warning: no reference WAVs in {ref_dir}; eval.py will need --no-fad in extras, "
            "or run preprocess with --jamendo-root.",
            file=sys.stderr,
        )

    storage = args.storage_root.resolve()
    runs_root = storage / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for spec in runs:
        name = spec.get("name")
        mc = spec.get("model_config")
        ck = spec.get("ckpt")
        if not name or not mc or not ck:
            raise ValueError(f"Each run needs name, model_config, ckpt; got {spec}")
        out_dir = runs_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        meta_path = out_dir / "sdd_run_meta.json"
        meta_path.write_text(
            json.dumps(
                {"name": name, "model_config": mc, "ckpt": ck, "num_samples": num_samples},
                indent=2,
            ),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--model-config",
            mc,
            "--ckpt",
            ck,
            "--prompts-json",
            str(prompts_json.resolve()),
            "--num-samples",
            str(num_samples),
            "--output-dir",
            str(out_dir),
        ]
        if use_fad:
            cmd.extend(["--reference-dir", str(ref_dir.resolve())])
        else:
            cmd.append("--no-fad")

        cmd.extend(eval_extra)
        if args.gpu:
            cmd.append("--gpu")

        print("\n" + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    if not args.no_summarize and not args.dry_run:
        summarize_runs(storage)


if __name__ == "__main__":
    main()
