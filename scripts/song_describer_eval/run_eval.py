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

By default each run **resumes** if ``<output>/generated/`` already has all expected
``*.wav`` + sidecar ``*.txt`` files (00000 … num_samples-1) and the saved manifest's
``num_samples`` matches the current preprocess. Those runs invoke eval.py with
``--skip-generation`` (FAD / CLAP only). If ``metrics.json`` already has the requested
FAD keys (non-null) and CLAP score when applicable, the eval subprocess is skipped
entirely. Use ``--reset`` to force regeneration and to force FAD/CLAP to be recomputed.

Any extra CLI tokens are forwarded to eval.py (e.g. --embedding clap-laion-music).
Use --gpu to add eval.py's --gpu flag for each run (CUDA_VISIBLE_DEVICES selects the GPU).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval.py"

DEFAULT_STORAGE = Path("/deepfreeze/pnlong/ambient/song_describer_eval")
DEFAULT_CACHE = DEFAULT_STORAGE / "cache"


def generation_artifacts_complete(generated_dir: Path, num_samples: int) -> bool:
    """True when eval.py-style outputs exist: 00000..(N-1).wav and matching .txt sidecars."""
    if num_samples <= 0:
        return False
    if not generated_dir.is_dir():
        return False
    for i in range(num_samples):
        base = generated_dir / f"{i:05d}"
        if not base.with_suffix(".wav").is_file():
            return False
        if not base.with_suffix(".txt").is_file():
            return False
    return True


def meta_num_samples_compatible(meta_path: Path, num_samples: int) -> bool:
    """False if a previous run recorded a different num_samples (preprocess changed)."""
    if not meta_path.is_file():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    old = meta.get("num_samples")
    if old is None:
        return True
    try:
        return int(old) == int(num_samples)
    except (TypeError, ValueError):
        return True


def _eval_extra_has_flag(extra: list[str], flag: str) -> bool:
    """True if ``flag`` appears as its own token or as ``--name=value``."""
    prefix = f"{flag}="
    return flag in extra or any(a.startswith(prefix) for a in extra)


def _embeddings_from_eval_extra(extra: list[str]) -> list[str]:
    """Tokens after ``--embedding`` / ``--embedding=``, mirroring eval.py defaults."""
    out: list[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok == "--embedding" and i + 1 < len(extra):
            out.append(extra[i + 1])
            i += 2
            continue
        if tok.startswith("--embedding="):
            out.append(tok.split("=", 1)[1])
            i += 1
            continue
        i += 1
    return out


def metrics_json_complete(
    metrics_path: Path,
    *,
    clap_expected: bool,
    fad_keys_expected: list[str] | None,
) -> bool:
    """True when saved metrics match what this eval command would compute."""
    if not metrics_path.is_file():
        return False
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    if clap_expected:
        if m.get("clap_score") is None:
            return False

    if fad_keys_expected is not None:
        fad = m.get("fad")
        if not isinstance(fad, dict):
            return False
        for k in fad_keys_expected:
            if fad.get(k) is None:
                return False

    return True


def _finite_floats(values: list[float]) -> list[float]:
    return [x for x in values if math.isfinite(x)]


def _tight_xlim_with_pad(ax, series: list[float], *, pad_frac: float = 0.08) -> None:
    """Set x-axis limits from data range + padding (do not pin at zero)."""
    vals = _finite_floats(series)
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    if lo == hi:
        pad = 0.05 * (abs(lo) + 1.0)
        ax.set_xlim(lo - pad, hi + pad)
        return
    span = hi - lo
    pad = max(span * pad_frac, 1e-9)
    ax.set_xlim(lo - pad, hi + pad)


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
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not installed; skipped summary.png (CSV only).", file=sys.stderr)
        return

    names = [r["run"] for r in rows]
    n = len(names)
    max_label = max((len(str(nm)) for nm in names), default=12)

    # Figure size: height from #runs (horizontal bars); width from label gutter + value panel(s).
    row_h_in = 0.34
    fig_h = max(3.6, 1.35 + n * row_h_in)
    left_gutter_in = 0.75 + 0.11 * min(max_label, 52)
    panel_w_in = max(3.45, fig_h * 0.40)
    fad_plot_keys = [k for k in all_keys if k.startswith("fad_")]
    share_y = bool(fad_plot_keys)
    if share_y:
        fig_w = left_gutter_in + 2.0 * panel_w_in + 0.4
    else:
        fig_w = left_gutter_in + panel_w_in + 1.35

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=share_y)
    ax_clap, ax_fad = axes[0], axes[1]

    def _xaxis_nice_ticks(ax, nbins: int = 6) -> None:
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=nbins, min_n_ticks=4, prune=None)
        )

    y = list(range(n))
    clap_bar_height = 0.65
    clap_vals = [
        float(r["clap_score"]) if r.get("clap_score") != "" else float("nan") for r in rows
    ]
    ax_clap.barh(y, clap_vals, color="steelblue", height=clap_bar_height)
    ax_clap.set_yticks(y)
    ax_clap.set_yticklabels(names)
    ax_clap.set_xlabel("CLAP score (↑ better)")
    ax_clap.set_title("Song Describer — text–audio alignment")
    _tight_xlim_with_pad(ax_clap, clap_vals)
    _xaxis_nice_ticks(ax_clap)

    if fad_plot_keys:
        n_emb = len(fad_plot_keys)
        # One FAD backend: match CLAP bar thickness. Several: split row height between series.
        if n_emb == 1:
            fad_slot_h = clap_bar_height
            fad_draw_h = clap_bar_height
        else:
            fad_slot_h = min(0.75 / n_emb, 0.2)
            fad_draw_h = fad_slot_h * 0.92
        for i, fk in enumerate(fad_plot_keys):
            if n_emb == 1:
                ypos = y
            else:
                offset = (i - (n_emb - 1) / 2.0) * fad_slot_h
                ypos = [yj + offset for yj in y]
            vals = [
                float(r[fk]) if r.get(fk) not in ("", None) else float("nan") for r in rows
            ]
            label = fk.replace("fad_", "")
            ax_fad.barh(ypos, vals, height=fad_draw_h, label=label)
        ax_fad.set_xlabel("FAD (↓ better)")
        ax_fad.set_title("Song Describer — distribution vs reference")
        ax_fad.legend(fontsize=8)
        # Shared y: only left panel shows run labels.
        ax_fad.tick_params(axis="y", left=False, labelleft=False)
        fad_all_vals: list[float] = []
        for fk in fad_plot_keys:
            for r in rows:
                v = r.get(fk)
                if v != "" and v is not None:
                    fad_all_vals.append(float(v))
        _tight_xlim_with_pad(ax_fad, fad_all_vals)
        _xaxis_nice_ticks(ax_fad)
    else:
        ax_fad.text(0.5, 0.5, "No FAD columns", ha="center", va="center", transform=ax_fad.transAxes)
        ax_fad.set_axis_off()

    ax_clap.invert_yaxis()
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore resume: regenerate audio, rerun FAD/CLAP, rewrite metrics",
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
        generated_dir = out_dir / "generated"

        meta_path = out_dir / "sdd_run_meta.json"
        meta_ok = meta_num_samples_compatible(meta_path, num_samples)
        meta_path.write_text(
            json.dumps(
                {"name": name, "model_config": mc, "ckpt": ck, "num_samples": num_samples},
                indent=2,
            ),
            encoding="utf-8",
        )

        skip_generation = (
            not args.reset
            and meta_ok
            and generation_artifacts_complete(generated_dir, num_samples)
        )

        clap_expected = not _eval_extra_has_flag(eval_extra, "--no-clap-score")
        if use_fad and not _eval_extra_has_flag(eval_extra, "--no-fad"):
            fad_keys_for_meta = _embeddings_from_eval_extra(eval_extra) or ["vggish"]
        else:
            fad_keys_for_meta = None

        metrics_path = out_dir / "metrics.json"
        metrics_ok = metrics_json_complete(
            metrics_path,
            clap_expected=clap_expected,
            fad_keys_expected=fad_keys_for_meta,
        )
        skip_eval = not args.reset and skip_generation and metrics_ok

        cmd: list[str] = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--output-dir",
            str(out_dir),
        ]
        if skip_eval:
            print(
                f"[resume] {name}: generation + metrics complete → skip eval.py",
                file=sys.stderr,
            )
            continue
        if skip_generation:
            cmd.append("--skip-generation")
            print(
                f"[resume] {name}: generation complete → eval.py --skip-generation",
                file=sys.stderr,
            )
        else:
            cmd.extend(
                [
                    "--model-config",
                    mc,
                    "--ckpt",
                    ck,
                    "--prompts-json",
                    str(prompts_json.resolve()),
                    "--num-samples",
                    str(num_samples),
                ]
            )
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
