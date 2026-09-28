"""The confirmatory analysis. Runs once, from the commit tagged confirmatory-v1.

    python scripts/confirmatory.py

Tests A and B per agent on analysis seeds (switch 0-99, control 1000-1099),
each at its calibrated switch_at / n_rounds / h / W / L. The in-context agent
is the primary hypothesis; the others are exploratory. Then the adaptation
ranking at the test opponent's standard structure (switch at 200 of 600).
Writes results/confirmatory/results.json.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path

import numpy as np

from regime.analysis import (
    AGENTS,
    ANALYSIS_CONTROL_SEEDS,
    ANALYSIS_SWITCH_SEEDS,
    DEFAULT_ROUNDS,
    DEFAULT_SWITCH_AT,
    Calibration,
    censored_quantile,
    displacement,
    lag_estimate,
    make_agent,
    peak_state_displacement,
    run_many,
    run_test_a,
    run_test_b,
    safe_recovery,
    signals,
)
from regime.metrics import excess_regret, moving_average

TAG = "confirmatory-v1"
OUT = Path("results/confirmatory/results.json")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def preflight() -> str:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the confirmatory analysis runs once")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes")
    head = git("rev-parse", "HEAD")
    if TAG not in git("tag", "--points-at", head).split():
        raise SystemExit(f"HEAD is not tagged {TAG}")
    return head


def load_cal(agent: str) -> Calibration:
    d = json.loads(Path(f"results/calibration/{agent}.json").read_text())
    return Calibration(**{f.name: d[f.name] for f in fields(Calibration)})


def regret_lag_signal(log) -> np.ndarray:
    """|Δ m(t)|: absolute first difference of 20-round windowed regret."""
    return np.abs(np.diff(moving_average(log.instant_regret, 20), prepend=np.nan))


def main() -> None:
    commit = preflight()
    workers = os.cpu_count()
    results = dict(commit=commit, tag=TAG, primary="in_context", agents={}, ranking={})

    for agent in AGENTS:
        cal = load_cal(agent)
        r = dict(role="primary" if agent == "in_context" else "exploratory", calibration=dict(
            t_w=cal.t_w, h=cal.h, W=cal.W, L=cal.L, switch_at=cal.switch_at, n_rounds=cal.n_rounds))
        if agent == "change_aware":
            r["note"] = "state contains output scores; Test B reported, not interpreted (AGENT_SPECS.md)"
        if not cal.runnable:
            r["tests"] = "not runnable: recovery censored above the 90th percentile"
            results["agents"][agent] = r
            continue

        sw = run_many(agent, "switch", ANALYSIS_SWITCH_SEEDS, cal.switch_at, cal.n_rounds, workers)
        ct = run_many(agent, "control", ANALYSIS_CONTROL_SEEDS, cal.switch_at, cal.n_rounds, workers)
        sig_sw = [signals(log, cal) for log in sw]
        sig_ct = [signals(log, cal) for log in ct]

        test_a = run_test_a([peak_state_displacement(s, cal.switch_at, cal.W) for s, _ in sig_sw],
                            [peak_state_displacement(s, cal.switch_at, cal.W) for s, _ in sig_ct])
        r["test_a"] = test_a
        lags = [lag_estimate(s, o, cal.switch_at, cal.W, cal.L) for s, o in sig_sw]
        r["test_b"] = run_test_b(lags) if test_a["passed"] else "not run: Test A did not pass (fixed sequence)"

        # Secondary, reported regardless of the gate and labeled as such.
        reg_lags = [lag_estimate(s, regret_lag_signal(log), cal.switch_at, cal.W, cal.L)
                    for (s, _), log in zip(sig_sw, sw)]
        r["secondary_regret_lag"] = run_test_b(reg_lags)
        if agent == "in_context":
            # Raw state = corrected state + the latest move's offset. Left unstandardized:
            # the calibrated statistics describe the corrected state, not the raw one.
            offsets = make_agent("in_context").move_offsets
            raw_lags = []
            for log in sw:
                raw_states = log.states + offsets[1, log.opp_actions]
                ds = displacement(raw_states, cal.h)
                do = displacement(log.outputs - log.outputs.mean(1, keepdims=True), cal.h)
                raw_lags.append(lag_estimate(ds, do, cal.switch_at, cal.W, cal.L))
            r["exploratory_raw_state_lag"] = run_test_b(raw_lags)
        results["agents"][agent] = r
        print(agent, json.dumps({k: v for k, v in r.items() if k in ("test_a", "calibration")}), flush=True)

    for agent in AGENTS:
        logs = run_many(agent, "switch", ANALYSIS_SWITCH_SEEDS, DEFAULT_SWITCH_AT, DEFAULT_ROUNDS, workers)
        times = [safe_recovery(log) for log in logs]
        ex = [excess_regret(log) for log in logs]
        results["ranking"][agent] = dict(
            recovery_median=censored_quantile(times, 0.5), recovery_censored=times.count(None),
            excess_regret_mean=float(np.mean(ex)), excess_regret_se=float(np.std(ex) / np.sqrt(len(ex))))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"saved {OUT} (commit {commit})")


if __name__ == "__main__":
    main()
