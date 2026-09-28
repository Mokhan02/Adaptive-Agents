"""Study 2: early detection (PREREG_EARLY_DETECTION.md).

    python scripts/early_detection.py power       # before the prereg is frozen; old calibration seeds only
    python scripts/early_detection.py run --dry-run
    python scripts/early_detection.py run         # once, from the tag early-detection-v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from regime.analysis import DRY_RUN_PAIR, TEST_PAIR, displacement
from regime.early_detection import (
    ALPHA_PRIMARY,
    ALPHA_SECONDARY,
    HELDOUT_CONTROL_SEEDS,
    SESOI,
    TEST_SWITCH_SEED_BASE,
    THRESHOLD_CONTROL_SEEDS,
    binomial_interval,
    detection_delay,
    false_alarm_rate,
    paired_differences,
    paired_test,
    run_max,
    threshold,
)
from scripts_common import load_cal

TAG = "early-detection-v1"
OUT = Path("results/early_detection")
N_ROUNDS = 300  # covers switch_at + W = 286; the model is causal, so later rounds can't matter


def episode(kind: str, pair, seed: int) -> dict:
    """One in-context episode; returns raw corrected residuals at layers 1 and 2, and output scores."""
    import torch

    from regime.agents.in_context import InContextAgent
    from regime.analysis import make_agent
    from regime.env import RegimeSwitchOpponent
    from regime.runner import run_episode

    torch.set_num_threads(1)
    cal = load_cal("in_context")
    base = make_agent("in_context")

    class TwoLayers(InContextAgent):
        def internal_state(self):
            r = self.residual_streams()  # latest-move corrected, per layer
            return np.concatenate([r[1], r[2]])

    agent = TwoLayers(base.model, temperature=base.temperature, move_offsets=base.move_offsets)
    first, second = pair
    opp = (RegimeSwitchOpponent(first, second, switch_at=cal.switch_at) if kind == "switch"
           else RegimeSwitchOpponent.no_switch(first, switch_at=cal.switch_at))
    log = run_episode(agent, opp, N_ROUNDS, seed)
    d = log.states.shape[1] // 2
    return dict(l1=log.states[:, :d], l2=log.states[:, d:], out=log.outputs)


def episodes(kind: str, seeds, pair) -> list[dict]:
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        return list(pool.map(partial(episode, kind, pair), seeds, chunksize=10))


def signal_series(runs, cal, l2_stats=None) -> dict[str, list[np.ndarray]]:
    mean1, std1 = np.array(cal.state_mean), np.array(cal.state_std)
    out = {
        "state_l1": [displacement((r["l1"] - mean1) / std1, cal.h) for r in runs],
        "output": [displacement(r["out"] - r["out"].mean(1, keepdims=True), cal.h) for r in runs],
    }
    if l2_stats is not None:
        mean2, std2 = l2_stats
        out["state_l2"] = [displacement((r["l2"] - mean2) / std2, cal.h) for r in runs]
    return out


# --- power and diagnostics (old calibration seeds only) ------------------------

def old_calibration_signals():
    from regime.analysis import CALIBRATION_CONTROL_SEEDS, CALIBRATION_SWITCH_SEEDS

    cal = load_cal("in_context")
    ctl = signal_series(episodes("control", CALIBRATION_CONTROL_SEEDS, TEST_PAIR), cal)
    sw = signal_series(episodes("switch", CALIBRATION_SWITCH_SEEDS, TEST_PAIR), cal)
    return cal, ctl, sw


def response_strengths(cal, ctl, sw) -> tuple[dict, dict]:
    """Per-signal threshold (alpha = 5%, from these controls) and typical response over threshold:
    (median switch-run max - median control-run max) / threshold."""
    a, W = cal.switch_at, cal.W
    keys = ("state_l1", "output")
    tau = {k: threshold([run_max(x, a, W) for x in ctl[k]], ALPHA_PRIMARY) for k in keys}
    resp = {k: float((np.median([run_max(x, a, W) for x in sw[k]]) - np.median([run_max(x, a, W) for x in ctl[k]])) / tau[k])
            for k in keys}
    return tau, resp


def run_diagnose() -> None:
    """What the first power attempt revealed: noise structure of each signal in controls, and detection
    delays of an identical-onset ramp at several strengths (controls plus planted ramps only)."""
    cal, ctl, sw = old_calibration_signals()
    a, W = cal.switch_at, cal.W
    tau, resp = response_strengths(cal, ctl, sw)
    rows = {}
    for k in ("state_l1", "output"):
        X = np.stack([x[a - W : a + W + 1] for x in ctl[k]])
        rows[k] = dict(mean=float(X.mean()), sd=float(X.std()), threshold=tau[k],
                       threshold_z=float((tau[k] - X.mean()) / X.std()),
                       lag1_autocorr=float(np.mean([np.corrcoef(x[:-1], x[1:])[0, 1] for x in X])),
                       response_over_threshold=resp[k])
    t = np.arange(N_ROUNDS)
    ramps = {}
    for c in (0.5, 1.0, 2.0):
        r = c * np.clip((t - (a + 5)) / 10, 0, 1)
        dl = {k: [detection_delay(x + tau[k] * r, tau[k], a, W) for x in ctl[k]] for k in ("state_l1", "output")}
        ramps[str(c)] = {k: dict(median_delay=float(np.median([d for d in v if d is not None])) if any(d is not None for d in v) else None,
                                 censored=sum(d is None for d in v)) for k, v in dl.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnose.json").write_text(json.dumps(dict(
        seeds_used="calibration controls 10100-10199, calibration switch runs 10000-10099",
        signals=rows, identical_onset_ramps=ramps), indent=2))
    print(json.dumps(rows, indent=1), json.dumps(ramps, indent=1))


def run_power(n_experiments: int = 200, sizes=(100, 200, 400, 800), n_boot: int = 2000, out_name: str = "power.json") -> None:
    """Planted power at the measured response strengths: each signal gets its own response over
    threshold; the null has identical onsets, the effect of interest a 2-round state lead."""
    cal, ctl, sw = old_calibration_signals()
    a, W = cal.switch_at, cal.W
    tau, resp = response_strengths(cal, ctl, sw)
    t = np.arange(N_ROUNDS)

    def experiment(n: int, lead: int, rng) -> dict:
        idx = rng.integers(0, len(ctl["output"]), size=n)
        d_s, d_o = [], []
        for i in idx:
            onset = a + int(rng.integers(0, 10))
            ramp = lambda o, c: c * np.clip((t - o) / 10, 0, 1)
            s = ctl["state_l1"][i] + tau["state_l1"] * ramp(onset - lead, resp["state_l1"])
            o = ctl["output"][i] + tau["output"] * ramp(onset, resp["output"])
            d_s.append(detection_delay(s, tau["state_l1"], a, W))
            d_o.append(detection_delay(o, tau["output"], a, W))
        return paired_test(paired_differences(d_s, d_o, W)[0], n_boot=n_boot)

    def dist(results) -> dict:
        out = {}
        for r in results:
            key = r["outcome"].split(":")[0]
            out[key] = out.get(key, 0) + 1 / len(results)
        return {k: round(v, 3) for k, v in sorted(out.items())}

    rng = np.random.default_rng(9_000_000)
    rows = {}
    for n in sizes:
        rows[n] = dict(outcomes_at_effect_of_interest=dist([experiment(n, SESOI, rng) for _ in range(n_experiments)]),
                       outcomes_at_null=dist([experiment(n, 0, rng) for _ in range(n_experiments)]))
        print(n, json.dumps(rows[n]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / out_name).write_text(json.dumps(dict(
        seeds_used="calibration controls 10100-10199, calibration switch runs 10000-10099 (old seeds only)",
        thresholds_for_power=tau, response_over_threshold=resp, experiments_per_cell=n_experiments, by_n=rows), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["power", "power-large", "diagnose"])
    args = ap.parse_args()
    if args.which == "power-large":
        run_power(n_experiments=100, sizes=(1600, 3200), n_boot=1000, out_name="power_large.json")
    else:
        {"power": run_power, "diagnose": run_diagnose}[args.which]()


if __name__ == "__main__":
    main()
