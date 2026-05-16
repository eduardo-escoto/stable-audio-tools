"""
Preprocess Song Describer (SDD) for scripts/eval.py.

Typical layout under --sdd-root:
  - song_describer.csv — caption_id, track_id, caption, is_valid_subset, …, path, duration
  - audio/             — MP3 tree; CSV `path` is relative to this dir (e.g. 34/1004034.mp3).
    Release copies often use *.2min.mp3 filenames; we resolve those automatically.

Use the project env when running (torchaudio for reference WAVs):
  mamba activate stable_audio_tools

If activate fails in a non-interactive shell:
  micromamba run -n stable_audio_tools python ...

Writes under --cache-dir:
  - prompts.json         — conditioning dicts for eval.py --prompts-json
  - manifest.json        — caption_id / track_id / path / caption (for traceability)
  - reference_wavs/      — cropped WAVs for FAD (--reference-dir), one file per unique track
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _truthy_subset(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("true", "1", "yes")


def _safe_relpath(path: str) -> str:
    """Filesystem-safe name for a relative path."""
    return re.sub(r"[^\w.\-]+", "_", path.replace("/", "__"))


def resolve_audio_mp3(audio_root: Path, csv_rel_path: str) -> Path:
    """Map CSV `path` to an on-disk MP3 under audio_root.

    Tries the path as given (e.g. 34/1004034.mp3), then the common SDD variant
    34/1004034.2min.mp3 when only the latter exists.
    """
    rel = csv_rel_path.strip().strip("/")
    if not rel:
        raise ValueError("empty path")

    direct = (audio_root / rel).resolve()
    try:
        direct.relative_to(audio_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes audio root: {csv_rel_path!r}") from exc

    if direct.is_file():
        return direct

    p = Path(rel)
    if p.suffix.lower() == ".mp3":
        stem = p.stem
        parent = p.parent
        two_min = (audio_root / parent / f"{stem}.2min.mp3").resolve()
        try:
            two_min.relative_to(audio_root.resolve())
        except ValueError as exc:
            raise ValueError(f"path escapes audio root: {csv_rel_path!r}") from exc
        if two_min.is_file():
            return two_min

    raise FileNotFoundError(
        f"No audio for CSV path {csv_rel_path!r} under {audio_root} "
        f"(tried {direct.name} and *.2min.mp3 variant)."
    )


def load_csv_rows(csv_path: Path, valid_only: bool) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if valid_only and not _truthy_subset(row.get("is_valid_subset")):
                continue
            rows.append(row)
    return rows


def build_manifest(rows: list[dict], seconds_total: float) -> tuple[list[dict], list[dict]]:
    """Return (manifest entries, prompts for eval)."""
    manifest = []
    prompts = []
    for row in rows:
        cap = (row.get("caption") or "").strip()
        if not cap:
            continue
        track_id = row.get("track_id", "")
        rel_audio = (row.get("path") or "").strip()
        manifest.append(
            {
                "caption_id": row.get("caption_id"),
                "track_id": track_id,
                "path": rel_audio,
                "caption": cap,
                "duration": row.get("duration"),
            }
        )
        prompts.append(
            {
                "prompt": cap,
                "seconds_start": 0,
                "seconds_total": seconds_total,
            }
        )
    return manifest, prompts


def mp3_to_cropped_wav(
    src: Path,
    dst: Path,
    crop_seconds: float,
    sample_rate: int,
) -> None:
    import torchaudio

    wav, sr = torchaudio.load(str(src))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    n_crop = int(crop_seconds * sr)
    if n_crop <= 0:
        raise ValueError("crop_seconds must be positive")
    if wav.shape[1] > n_crop:
        wav = wav[:, :n_crop]
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), wav, sample_rate)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sdd-root",
        type=Path,
        default=Path("/deepfreeze/pnlong/ambient/song_describer"),
        help="Directory containing song_describer.csv",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/deepfreeze/pnlong/ambient/song_describer_eval/cache"),
        help="Output directory for prompts + manifest + reference_wavs/",
    )
    p.add_argument(
        "--valid-only",
        action="store_true",
        default=True,
        help="Keep only rows with is_valid_subset=True (default: on)",
    )
    p.add_argument(
        "--all-rows",
        action="store_true",
        help="Use every CSV row (overrides --valid-only)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Keep only first N rows after filtering (0 = no limit)",
    )
    p.add_argument(
        "--seconds-total",
        type=float,
        default=12.0,
        help="seconds_total in prompts and crop length for reference WAVs",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate for reference WAVs",
    )
    p.add_argument(
        "--jamendo-root",
        type=Path,
        default=None,
        help="Directory containing the CSV `path` segments (usually <sdd-root>/audio). "
        "Default: <sdd-root>/audio if that directory exists, else unset (no reference WAVs).",
    )
    p.add_argument(
        "--skip-reference",
        action="store_true",
        help="Do not create reference_wavs/ even if --jamendo-root is set",
    )
    args = p.parse_args()

    sdd_root = args.sdd_root.resolve()
    default_audio = sdd_root / "audio"
    if args.jamendo_root is None and default_audio.is_dir():
        args.jamendo_root = default_audio

    valid_only = not args.all_rows
    csv_path = sdd_root / "song_describer.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    rows = load_csv_rows(csv_path, valid_only=valid_only)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    manifest, prompts = build_manifest(rows, args.seconds_total)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.cache_dir / "prompts.json"
    manifest_path = args.cache_dir / "manifest.json"
    prompts_path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts → {prompts_path}")
    print(f"Wrote manifest      → {manifest_path}")

    if args.skip_reference:
        print("--skip-reference: skipped reference_wavs/.")
        return

    if args.jamendo_root is None:
        print(
            "No audio root (<sdd-root>/audio missing and --jamendo-root not set): "
            "skipped reference_wavs/ (FAD will need --no-fad unless you add WAVs)."
        )
        return

    jamendo = args.jamendo_root.resolve()
    ref_dir = args.cache_dir / "reference_wavs"
    seen_track: dict[str, Path] = {}

    for m in manifest:
        rel = m["path"]
        if not rel:
            continue
        tid = str(m["track_id"])
        if tid in seen_track:
            continue
        src = resolve_audio_mp3(jamendo, rel)
        # One WAV per track; multiple captions share the same reference clip.
        dst = ref_dir / f"{tid}_{_safe_relpath(rel)}.wav"
        mp3_to_cropped_wav(src, dst, args.seconds_total, args.sample_rate)
        seen_track[tid] = dst

    print(f"Wrote {len(seen_track)} reference WAVs → {ref_dir}")


if __name__ == "__main__":
    main()
