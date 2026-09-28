"""Hyperparameter sweep for the in-context agent (GPU pretraining, CPU data).

    python scripts/tune_icl.py --parallel 8

Protocol: AGENT_SPECS.md, amendment "tuning protocol". Each of the 30
configurations (lr, steps, tau) gets its own pretraining run in
<runs-dir>/cfgNN, and its best.pt is scored on the 200 tuning episodes.
Batch generation runs on the CPU and dominates step time on a GPU, so
several configurations train at once as separate processes.

Resumable: finished configurations are recorded in <out>/in_context.jsonl
and skipped, and an interrupted pretraining run resumes from its latest.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from regime.agents.in_context import load_model
from regime.tuning import N_TUNING_EPISODES, sample_configs, score_in_context_fast

LOCK = threading.Lock()


def run_config_safe(idx: int, config: dict, args) -> dict | None:
    """One failed configuration is reported and skipped, not fatal to the sweep.
    Rerunning the sweep retries it."""
    try:
        return run_config(idx, config, args)
    except Exception as e:  # noqa: BLE001
        with LOCK:
            print(f"[{idx:>2}] FAILED: {e!r} (see {args.runs_dir / f'cfg{idx:02d}.out'})", flush=True)
        return None


def run_config(idx: int, config: dict, args) -> dict:
    run_dir = args.runs_dir / f"cfg{idx:02d}"
    cmd = [sys.executable, "scripts/pretrain_icl.py", "--run-dir", str(run_dir), "--steps", str(config["steps"]),
           "--lr", repr(config["lr"]), "--threads", "1", "--device", args.device]
    t0 = time.time()
    with open(args.runs_dir / f"cfg{idx:02d}.out", "a") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=os.environ | {"PYTHONPATH": "src"})
    model = load_model(run_dir / "best.pt")
    scores = [score_in_context_fast(model, config["temperature"], i) for i in range(N_TUNING_EPISODES)]
    rec = dict(index=idx, config=config, score=float(np.mean(scores)),
               se=float(np.std(scores) / np.sqrt(len(scores))), sec=round(time.time() - t0, 1))
    with LOCK:
        with open(args.out / "in_context.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{idx:>2}] score {rec['score']:7.2f} ± {rec['se']:.2f}  {config}  ({rec['sec']}s)", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=8, help="configurations training at once")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/icl_sweep"))
    ap.add_argument("--out", type=Path, default=Path("results/tuning"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "in_context.jsonl"
    done = {json.loads(line)["index"] for line in open(path)} if path.exists() else set()

    configs = sample_configs("in_context")
    todo = [(i, c) for i, c in enumerate(configs) if i not in done]
    # Longest runs first, so the last wave isn't one 20k-step run on its own.
    todo.sort(key=lambda ic: -ic[1]["steps"])
    print(f"{len(todo)} configurations to run on {args.device}, {args.parallel} at a time", flush=True)
    with ThreadPoolExecutor(args.parallel) as pool:
        outcomes = list(pool.map(lambda ic: run_config_safe(*ic, args), todo))

    failed = [i for (i, _), r in zip(todo, outcomes) if r is None]
    if failed:
        print(f"{len(failed)} configuration(s) failed: {failed}. Rerun to retry them.")
    results = [json.loads(line) for line in open(path)]
    if len(results) < len(configs):
        print(f"only {len(results)}/{len(configs)} configurations scored; the winner is provisional until all are")
    best = min(results, key=lambda r: (r["score"], r["index"]))
    best_dir = args.runs_dir / f"cfg{best['index']:02d}"
    print(f"best: [{best['index']}] {best['score']:.2f}  {best['config']}  -> {best_dir}")


if __name__ == "__main__":
    main()
