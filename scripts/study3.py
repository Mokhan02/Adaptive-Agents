"""Study 3 (ANALYSIS_PLAN.md, "Study 3").

    python scripts/study3.py calibrate   # reference agents only; writes results/study3/calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from regime import study3 as s3

OUT = Path("results/study3")


def _ref_run(name: str, window: int, n_rounds: int, seed: int) -> np.ndarray:
    from regime.env import FictitiousPlayOpponent
    from regime.runner import run_episode

    agent = s3.reference_agents()[name]()
    return run_episode(agent, FictitiousPlayOpponent(window, s3.EPS), n_rounds, seed).expected_rewards


def ref_runs(name: str, window: int, seeds, n_rounds: int = s3.N_ROUNDS) -> np.ndarray:
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        return np.stack(list(pool.map(partial(_ref_run, name, window, n_rounds), seeds, chunksize=5)))


def calibrate() -> None:
    out = OUT / "calibration.json"
    if out.exists():
        raise SystemExit(f"{out} exists; calibration runs once")
    names = list(s3.reference_agents())
    lo, hi = s3.SELECTION_WINDOW

    selection, chosen = {}, None
    for M in s3.M_GRID:
        scores = {n: ref_runs(n, M, s3.M_SEEDS)[:, lo:hi].mean(1) for n in names}
        means = {n: float(v.mean()) for n, v in scores.items()}
        ses = {n: float(v.std() / np.sqrt(len(v))) for n, v in scores.items()}
        ok = s3.separated(means, ses)
        selection[M] = dict(means=means, ses=ses, all_pairs_separated=ok)
        print(f"M={M:>2}: " + "  ".join(f"{n} {means[n]:+.3f}±{ses[n]:.3f}" for n in names) + f"  separated={ok}", flush=True)
        if ok and chosen is None:
            chosen = M
            break
    rule_failed = chosen is None
    M = s3.M_FALLBACK if rule_failed else chosen

    tw_runs = {n: ref_runs(n, M, s3.TW_SEEDS) for n in names}
    tw = {n: s3.settle_round(np.mean([s3.trailing_mean(r) for r in runs], axis=0)) for n, runs in tw_runs.items()}
    T_w = max(tw.values())
    n_rounds = s3.N_ROUNDS if T_w <= 700 else T_w + 300
    sigmas = {n: float(runs[:, T_w:].mean(1).std()) for n, runs in tw_runs.items()}
    sigma_ref = max(sigmas.values())
    N, underpowered = s3.choose_n(sigma_ref)
    power_table = {n: float(s3.required_n(sigma_ref * s3.INFLATION)) for n in ["required_n_per_agent"]}

    result = dict(
        M=M, M_rule_failed=rule_failed, selection=selection, eps=s3.EPS,
        T_w=T_w, T_w_by_reference=tw, n_rounds=n_rounds,
        score_sd_by_reference=sigmas, sigma_ref=sigma_ref, inflation=s3.INFLATION,
        sesoi=s3.SESOI, N=N, underpowered=underpowered, **power_table,
        seeds=dict(M_selection=[s3.M_SEEDS.start, s3.M_SEEDS.stop - 1], T_w=[s3.TW_SEEDS.start, s3.TW_SEEDS.stop - 1]),
        note="reference agents only; the three frozen agents were not run",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"M={M} (rule failed: {rule_failed})  T_w={T_w} {tw}  n_rounds={n_rounds}  sigma_ref={sigma_ref:.4f}  "
          f"required n={result['required_n_per_agent']:.0f} -> N={N}{' (underpowered)' if underpowered else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["calibrate"])
    args = ap.parse_args()
    {"calibrate": calibrate}[args.which]()


if __name__ == "__main__":
    main()
