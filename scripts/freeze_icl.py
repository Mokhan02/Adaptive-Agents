"""Freeze the in-context agent: bundle best.pt with its move offsets.

    python scripts/freeze_icl.py --run-dir runs/icl --out models/icl_frozen.pt

The move offsets are fit to this exact checkpoint's state distribution, so
the weights and offsets are saved as one file, tied together by a hash of
the weights. `load_frozen` refuses a bundle whose weights no longer match.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from regime.agents.in_context import estimate_move_offsets, load_model, save_frozen


def cpu_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    log = [json.loads(line) for line in open(args.run_dir / "log.jsonl")]
    best = min(log, key=lambda r: r["val_ce"])
    model = load_model(args.run_dir / "best.pt")
    offsets = estimate_move_offsets(model)
    train_threads = torch.load(args.run_dir / "latest.pt", map_location="cpu")["args"]["threads"] or "default"
    meta = dict(
        step=best["step"],
        val_ce=best["val_ce"],
        val_excess=best["val_excess"],
        total_steps=log[-1]["step"],
        train_seconds=log[-1]["sec"],
        hardware=f"{cpu_name()}, {train_threads} torch threads / {platform.platform()} / torch {torch.__version__}",
        frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    digest = save_frozen(model, offsets, meta, args.out)
    print(json.dumps(meta | {"weights_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
