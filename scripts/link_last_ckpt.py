#!/usr/bin/env python3
"""
link_last_ckpt.py — point every training run's ``last.ckpt`` at the newest checkpoint.

**What it is for:** Lightning/Stable Audio runs are stored as ``<runs-root>/<run_id>/``.
Each run saves checkpoints under ``checkpoints/`` with names like
``epoch=9-step=20000.ckpt``. Downstream (eval, demos) often expect
``checkpoints/last.ckpt``. This script scans each ``run_id``, finds the file with
the **largest** ``step=…`` in the filename, and makes ``last.ckpt`` refer to it.

**Hard links vs symlinks:** By default ``last.ckpt`` is a **hard link** (``os.link``)
to that file, so it behaves as a normal file when opened. That matters on some
shared mounts (e.g. parts of ``/deepfreeze``) where **symbolic links are broken**
(they appear in ``ls -l`` but ``readlink`` / ``open`` fail). Use ``--symlink`` on
normal local disks if you prefer a relative symlink.

**Bookkeeping:** After a successful link, the script writes
``checkpoints/.last_ckpt_record.json`` (target filename + step). The next run
**skips** that folder if nothing changed, so re-running the script is cheap.
Use ``--reset`` to ignore that record and re-create ``last.ckpt`` for every run id
(e.g. you want a full refresh or suspect stale state).

**Flags:** ``--runs-root``, ``--dry-run``, ``--symlink``, ``--reset``.

Example:
    mamba activate stable_audio_tools
    python scripts/link_last_ckpt.py
    python scripts/link_last_ckpt.py --runs-root /deepfreeze/pnlong/ambient/runs/ambient --dry-run
    python scripts/link_last_ckpt.py --reset

Requires tqdm (already in stable-audio-tools setup.py).
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm


STEP_PATTERN = re.compile(r"^epoch=\d+-step=(\d+)\.ckpt$")
RECORD_NAME = ".last_ckpt_record.json"


def find_highest_step_ckpt(checkpoints_dir: Path) -> Path | None:
    best_path: Path | None = None
    best_step = -1
    for p in checkpoints_dir.iterdir():
        if not p.is_file() and not p.is_symlink():
            continue
        if p.name == "last.ckpt":
            continue
        m = STEP_PATTERN.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = p
    return best_path


def _last_ckpt_readable(link_path: Path) -> bool:
    try:
        with link_path.open("rb") as f:
            f.read(1)
        return True
    except OSError:
        return False


def _load_record(checkpoints_dir: Path) -> dict | None:
    record_path = checkpoints_dir / RECORD_NAME
    if not record_path.is_file():
        return None
    try:
        return json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _save_record(checkpoints_dir: Path, target_name: str, step: int) -> None:
    record_path = checkpoints_dir / RECORD_NAME
    record_path.write_text(
        json.dumps({"target": target_name, "step": step}, indent=2),
        encoding="utf-8",
    )


def _remove_last(link_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()


def link_last(
    checkpoints_dir: Path,
    dry_run: bool,
    use_symlink: bool,
    reset: bool,
) -> tuple[str, str | None]:
    """Return (status, detail). status one of ok, skip, error."""
    best = find_highest_step_ckpt(checkpoints_dir)
    if best is None:
        return "skip", "no matching epoch=*-step=*.ckpt files"

    m = STEP_PATTERN.match(best.name)
    if not m:
        return "skip", "internal: best file name unmatched"
    best_step = int(m.group(1))

    link_path = checkpoints_dir / "last.ckpt"
    if not reset:
        record = _load_record(checkpoints_dir)
        if (
            record
            and record.get("target") == best.name
            and record.get("step") == best_step
            and _last_ckpt_readable(link_path)
        ):
            return "skip", f"last.ckpt already OK -> {best.name}"

    strategy = "symlink" if use_symlink else "hardlink"
    if dry_run:
        return "ok", f"would replace last.ckpt via {strategy} -> {best.name}"

    _remove_last(link_path)

    if use_symlink:
        link_path.symlink_to(best.name)
        _save_record(checkpoints_dir, best.name, best_step)
        return "ok", f"last.ckpt symlink -> {best.name}"

    try:
        os.link(best, link_path)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            return (
                "error",
                f"hardlink failed (EXDEV); retry with --symlink or same-device checkpoint: {exc}",
            )
        return "error", f"hardlink failed: {exc}"

    _save_record(checkpoints_dir, best.name, best_step)
    return "ok", f"last.ckpt hardlinked -> {best.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/deepfreeze/pnlong/ambient/runs/ambient"),
        help="Directory containing per-run id subfolders",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use a relative symlink instead of a hard link (for local disks; not for broken symlink mounts)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not create or change last.ckpt",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore .last_ckpt_record.json and refresh last.ckpt for every run id",
    )
    args = parser.parse_args()

    root = args.runs_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    candidates = sorted([p for p in root.iterdir() if p.is_dir()])
    if not candidates:
        print(f"No subdirectories under {root}", file=sys.stderr)
        sys.exit(1)

    stats = {"ok": 0, "skip": 0, "error": 0}

    for run_dir in tqdm(candidates, desc="Run ids", unit="id"):
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            stats["skip"] += 1
            tqdm.write(f"[skip] {run_dir.name}: no checkpoints/ directory")
            continue

        status, detail = link_last(ckpt_dir, args.dry_run, args.symlink, args.reset)
        stats[status] += 1
        prefix = {"ok": "[ok]", "skip": "[skip]", "error": "[err]"}[status]
        tqdm.write(f"{prefix} {run_dir.name}: {detail}")

    print(
        f"Done: ok={stats['ok']} skip={stats['skip']} error={stats['error']} "
        f"({'dry-run' if args.dry_run else 'updated'})"
    )
    if stats["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
