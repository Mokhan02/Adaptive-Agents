"""The confirmatory analysis. Runs once, from the commit tagged confirmatory-v1.

    python scripts/confirmatory.py              # the real run: A -> B, seeds 0-99
    python scripts/confirmatory.py --dry-run    # pipeline check on a non-A/B pair, 6 seeds

Tests A and B per agent on analysis seeds (switch 0-99, control 1000-1099),
each at its calibrated switch_at / n_rounds / h / W / L. The in-context agent
is the primary hypothesis; the others are exploratory. Then the adaptation
ranking: headline at the test opponent's standard structure (switch at 200 of
600), plus a sensitivity variant at each agent's own switch_at. Writes
results/confirmatory/results.json, unedited, whatever the outcome.
"""

from __future__ import annotations

import argparse
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
    DRY_RUN_PAIR,
    TEST_PAIR,
    Calibration,
    censored_quantile,
    displacement,
    lag_estimate,
    make_agent,
    peak_state_displacement,
    recovery_outcome,
    run_many,
    run_test_a,
    run_test_b,
    signals,
    xcorr_curve,
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
    """|delta m(t)|: absolute first difference of 20-round windowed regret."""
    return np.abs(np.diff(moving_average(log.instant_regret, 20), prepend=np.nan))


def mean_curve(curves: list[dict[int, float]]) -> dict[str, float]:
    return {str(k): float(np.mean([c[k] for c in curves])) for k in curves[0]}


def outcome(test_a: dict, test_b) -> str:
    """The outcome mapping in ANALYSIS_PLAN.md."""
    if not test_a["passed"]:
        return "Test A failed: state does not detectably respond; setup under-powered (not a lead/lag null)"
    if not test_b["significant"]:
        lo, hi = test_b["median_ci95"]
        return f"No separable lead/lag; median lag {test_b['median_lag']:g}, 95% CI [{lo:g}, {hi:g}]"
    med = test_b["median_lag"]
    if med > 0:
        return f"Representation leads behavior by a median {med:g} rounds"
    if med < 0:
        return f"Representation lags behavior by a median {-med:g} rounds"
    direction = "state first" if test_b["n_positive"] > test_b["n_negative"] else "behavior first"
    return f"Sign test significant ({direction} among nonzero lags) but median lag 0: lead below resolution"


def ranking_row(logs) -> dict:
    outs = [recovery_outcome(log) for log in logs]
    times = [t for t, _ in outs]
    ex = [excess_regret(log) for log in logs]
    return dict(
        recovery_median=censored_quantile(times, 0.5),
        n_recovered=sum(r == "recovered" for _, r in outs),
        n_not_recovered=sum(r == "not_recovered" for _, r in outs),
        n_no_pre_switch_edge=sum(r == "no_pre_switch_edge" for _, r in outs),
        excess_regret_mean=float(np.mean(ex)),
        excess_regret_se=float(np.std(ex) / np.sqrt(len(ex))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="non-A/B pair, few seeds, scratch output; never the real run")
    ap.add_argument("--dry-out", type=Path, default=Path("results/dry_run/results.json"))
    args = ap.parse_args()

    if args.dry_run:
        commit, pair, out = git("rev-parse", "HEAD") + " (dry run)", DRY_RUN_PAIR, args.dry_out
        sw_seeds, ct_seeds = range(6), range(1000, 1006)
    else:
        commit, pair, out = preflight(), TEST_PAIR, OUT
        sw_seeds, ct_seeds = ANALYSIS_SWITCH_SEEDS, ANALYSIS_CONTROL_SEEDS
    workers = os.cpu_count()
    results = dict(commit=commit, tag=TAG, dry_run=args.dry_run, primary="in_context",
                   pair=[p.tolist() for p in pair], agents={}, ranking={})

    own_switch_logs = {}
    for agent in AGENTS:
        cal = load_cal(agent)
        r = dict(role="primary" if agent == "in_context" else "exploratory", calibration=dict(
            t_w=cal.t_w, h=cal.h, W=cal.W, L=cal.L, switch_at=cal.switch_at, n_rounds=cal.n_rounds))
        if agent == "change_aware":
            r["note"] = "state contains output scores; Test B reported, not interpreted (AGENT_SPECS.md)"
        if not cal.runnable:
            r["outcome"] = "not runnable: recovery censored above the 90th percentile"
            results["agents"][agent] = r
            continue

        sw = run_many(agent, "switch", sw_seeds, cal.switch_at, cal.n_rounds, workers, pair)
        ct = run_many(agent, "control", ct_seeds, cal.switch_at, cal.n_rounds, workers, pair)
        own_switch_logs[agent] = sw
        sig_sw = [signals(log, cal) for log in sw]
        sig_ct = [signals(log, cal) for log in ct]

        test_a = run_test_a([peak_state_displacement(s, cal.switch_at, cal.W) for s, _ in sig_sw],
                            [peak_state_displacement(s, cal.switch_at, cal.W) for s, _ in sig_ct])
        r["test_a"] = test_a
        if test_a["passed"]:
            test_b = run_test_b([lag_estimate(s, o, cal.switch_at, cal.W, cal.L) for s, o in sig_sw])
            r["test_b"] = test_b
            r["test_b_mean_xcorr"] = mean_curve([xcorr_curve(s, o, cal.switch_at, cal.W, cal.L) for s, o in sig_sw])
        else:
            test_b = None
            r["test_b"] = "not run: Test A did not pass (fixed sequence)"
        r["outcome"] = outcome(test_a, test_b)

        # Secondary: reported regardless of the gate; not evidence for the primary result.
        reg = [(s, regret_lag_signal(log)) for (s, _), log in zip(sig_sw, sw)]
        r["secondary_regret_lag"] = run_test_b([lag_estimate(s, g, cal.switch_at, cal.W, cal.L) for s, g in reg])
        r["secondary_regret_lag_mean_xcorr"] = mean_curve([xcorr_curve(s, g, cal.switch_at, cal.W, cal.L) for s, g in reg])

        if agent == "in_context":
            # Raw state = corrected state + the latest move's offset. Left unstandardized:
            # the calibrated statistics describe the corrected state, not the raw one.
            offsets = make_agent("in_context").move_offsets
            raw_lags = []
            for log in sw:
                ds = displacement(log.states + offsets[1, log.opp_actions], cal.h)
                do = displacement(log.outputs - log.outputs.mean(1, keepdims=True), cal.h)
                raw_lags.append(lag_estimate(ds, do, cal.switch_at, cal.W, cal.L))
            r["exploratory_raw_state_lag"] = run_test_b(raw_lags)

        results["agents"][agent] = r
        print(f"{agent}: {r['outcome']}", flush=True)

    for agent in AGENTS:
        common = run_many(agent, "switch", sw_seeds, DEFAULT_SWITCH_AT, DEFAULT_ROUNDS, workers, pair)
        results["ranking"][agent] = dict(
            headline_common_switch_200=ranking_row(common),
            sensitivity_own_switch=ranking_row(own_switch_logs[agent]) if agent in own_switch_logs else None,
        )

    def order(variant: str) -> list[str]:
        rows = {a: v[variant] for a, v in results["ranking"].items() if v[variant]}
        return sorted(rows, key=lambda a: rows[a]["excess_regret_mean"])

    results["ranking_order_by_excess_regret"] = dict(
        headline=order("headline_common_switch_200"), sensitivity=order("sensitivity_own_switch"))
    results["ranking_orders_agree"] = (results["ranking_order_by_excess_regret"]["headline"]
                                       == results["ranking_order_by_excess_regret"]["sensitivity"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out} (commit {commit})")


if __name__ == "__main__":
    main()
