"""Calibrate one agent on A -> B (seeds 10000+).

    python scripts/calibrate.py --agent in_context

Derives T_w, h, W, L, switch_at, n_rounds and the state standardization
(ANALYSIS_PLAN.md amendments). Never computes Tests A or B. Writes
results/calibration/<agent>.json and a recovery-time summary.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from regime.analysis import (
    AGENTS,
    CALIBRATION_CONTROL_SEEDS,
    CALIBRATION_SWITCH_SEEDS,
    DEFAULT_ROUNDS,
    DEFAULT_SWITCH_AT,
    calibrate,
    run_many,
    safe_recovery,
    save,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=AGENTS, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=Path("results/calibration"))
    args = ap.parse_args()

    out = args.out / f"{args.agent}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; calibration runs once per agent")

    sw = run_many(args.agent, "switch", CALIBRATION_SWITCH_SEEDS, DEFAULT_SWITCH_AT, DEFAULT_ROUNDS, args.workers)
    ct = run_many(args.agent, "control", CALIBRATION_CONTROL_SEEDS, DEFAULT_SWITCH_AT, DEFAULT_ROUNDS, args.workers)
    cal = calibrate(args.agent, sw, ct)
    save(cal, out)

    times = [safe_recovery(log) for log in sw]
    obs = [t for t in times if t is not None]
    print(f"{args.agent}: recovery times (n=100): {len(obs)} observed, {cal.n_censored} censored")
    if obs:
        print(f"  observed: min {min(obs)}, median {np.median(obs):.0f}, max {max(obs)}")
    print(f"  censored median {cal.median_recovery}, p90 {cal.p90_recovery}")
    print(f"  T_w={cal.t_w}  h={cal.h}  W={cal.W}  L={cal.L}  switch_at={cal.switch_at}  n_rounds={cal.n_rounds}  "
          f"runnable={cal.runnable}")
    if cal.runnable:
        assert cal.switch_at - cal.W >= cal.t_w + cal.h and cal.switch_at + cal.W < cal.n_rounds
        print(f"  window [{cal.switch_at - cal.W}, {cal.switch_at + cal.W}] lies after warm-up "
              f"(T_w + h = {cal.t_w + cal.h}) and inside the episode (n_rounds = {cal.n_rounds})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
