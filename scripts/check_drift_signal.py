"""Early check that the in-context agent's drift signal can detect a switch.

    python scripts/check_drift_signal.py --model runs/icl/best.pt
    python scripts/check_drift_signal.py --model random   # untrained baseline

Uses only pretraining-distribution opponents, never A or B (AGENT_SPECS.md
forbids evaluating the model on the test strategies before its checkpoint is
frozen). Each switch run moves between two allowed strategies about as far
apart as A and B (TV 0.35-0.45), and each is paired with a no-switch control on
its first strategy. Detection rate = share of switch runs whose peak
displacement in the 60 rounds after the switch exceeds the 95th percentile of
the controls' peaks. Chance is 5%.

Compare against the untrained network: any function of the last 50 moves
shifts when the move mix shifts, so detection alone does not mean the model
learned anything.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from regime.agents.in_context import InContextAgent, ModelConfig, MoveTransformer, estimate_move_offsets, load_model
from regime.env import RegimeSwitchOpponent
from regime.pretrain import _sample_strategy, tv
from regime.runner import run_episode

DIAGNOSTIC_SEED_BASE = 3_000_000  # disjoint from analysis, calibration, pretraining and validation seeds
SWITCH_AT, N_ROUNDS, POST = 200, 320, 60


def displacement(x: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[h:] = np.linalg.norm(x[h:] - x[:-h], axis=1)
    return out


def sample_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = _sample_strategy(rng)
    while True:
        q = _sample_strategy(rng)
        if 0.35 <= tv(p, q) <= 0.45:
            return p, q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint path, or 'random' for an untrained network")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 20])
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.model == "random":
        torch.manual_seed(0)
        model = MoveTransformer(ModelConfig()).eval()
    else:
        model = load_model(args.model)

    offsets = estimate_move_offsets(model)
    signals = {"switch": [], "control": []}
    for i in range(args.pairs):
        p, q = sample_pair(np.random.default_rng(DIAGNOSTIC_SEED_BASE + i))
        for kind, opp in [
            ("switch", RegimeSwitchOpponent(p, q, switch_at=SWITCH_AT)),
            ("control", RegimeSwitchOpponent.no_switch(p, switch_at=SWITCH_AT)),
        ]:
            agent = InContextAgent(model, move_offsets=offsets)
            log = run_episode(agent, opp, N_ROUNDS, seed=DIAGNOSTIC_SEED_BASE + i)
            raw = log.states + offsets[1, log.opp_actions]
            signals[kind].append({"state, raw": raw, "state, corrected": log.states, "output scores": log.outputs})

    print(f"model={args.model}  pairs={args.pairs}  (chance = 5%)")
    for sig in ["state, raw", "state, corrected", "output scores"]:
        rates = []
        for h in args.horizons:
            peak = {
                kind: np.array([np.nanmax(displacement(r[sig], h)[SWITCH_AT : SWITCH_AT + POST]) for r in runs])
                for kind, runs in signals.items()
            }
            thr = np.percentile(peak["control"], 95)
            rates.append(f"h={h}: {np.mean(peak['switch'] > thr):.0%}")
        print(f"  {sig:<17} " + "   ".join(rates))


if __name__ == "__main__":
    main()
