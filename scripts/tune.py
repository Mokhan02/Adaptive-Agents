"""Hyperparameter sweep for the fine-tuning and RL agents (CPU, many processes).

    python scripts/tune.py --agent change_aware --workers 4 --out results/tuning

Protocol: AGENT_SPECS.md, amendment "tuning protocol". Each of the 30
configurations plays the same 200 tuning episodes; the score is mean
oracle-normalized excess regret. Results are appended per configuration to
<out>/<agent>.jsonl, so an interrupted sweep resumes where it stopped.

The in-context sweep needs a pretraining run per configuration and is run
separately (see scripts/pretrain_icl.py).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from regime.agents.online import ChangeAwareAgent, FineTuneAgent
from regime.tuning import N_TUNING_EPISODES, sample_configs, score_episode

AGENTS = {"fine_tune": FineTuneAgent, "change_aware": ChangeAwareAgent}


def _score(agent: str, config: dict, i: int) -> float:
    import torch

    torch.set_num_threads(1)  # one process per core; batch-1 updates don't benefit from threads
    return score_episode(partial(AGENTS[agent], **config), i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=sorted(AGENTS), required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=Path("results/tuning"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.agent}.jsonl"
    done = {json.loads(line)["index"] for line in open(path)} if path.exists() else set()

    configs = sample_configs(args.agent)
    with ProcessPoolExecutor(args.workers) as pool:
        for idx, config in enumerate(configs):
            if idx in done:
                continue
            t0 = time.time()
            scores = list(pool.map(partial(_score, args.agent, config), range(N_TUNING_EPISODES), chunksize=10))
            rec = dict(index=idx, config=config, score=float(np.mean(scores)),
                       se=float(np.std(scores) / np.sqrt(len(scores))), sec=round(time.time() - t0, 1))
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{idx:>2}] score {rec['score']:7.2f} ± {rec['se']:.2f}  {config}  ({rec['sec']}s)", flush=True)

    results = [json.loads(line) for line in open(path)]
    best = min(results, key=lambda r: (r["score"], r["index"]))
    print(f"best: [{best['index']}] {best['score']:.2f}  {best['config']}")


if __name__ == "__main__":
    main()
