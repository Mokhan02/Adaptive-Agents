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


def run_bias() -> None:
    """Where does the planted-null "state first" bias come from? Split the null deltas by pair type."""
    cal, ctl, sw = old_calibration_signals()
    a, W = cal.switch_at, cal.W
    tau, resp = response_strengths(cal, ctl, sw)
    t = np.arange(N_ROUNDS)
    rng = np.random.default_rng(9_100_000)
    d_s, d_o = [], []
    for i in rng.integers(0, len(ctl["output"]), size=20_000):
        onset = a + int(rng.integers(0, 10))
        ramp = lambda c: c * np.clip((t - onset) / 10, 0, 1)
        d_s.append(detection_delay(ctl["state_l1"][i] + tau["state_l1"] * ramp(resp["state_l1"]), tau["state_l1"], a, W))
        d_o.append(detection_delay(ctl["output"][i] + tau["output"] * ramp(resp["output"]), tau["output"], a, W))
    rows = {}
    for kind in ("both_detected", "state_only_censored", "output_only_censored", "both_censored"):
        sel = [(s_, o_) for s_, o_ in zip(d_s, d_o) if
               {"both_detected": s_ is not None and o_ is not None, "state_only_censored": s_ is None and o_ is not None,
                "output_only_censored": s_ is not None and o_ is None, "both_censored": s_ is None and o_ is None}[kind]]
        deltas = paired_differences([x for x, _ in sel], [y for _, y in sel], W)[0]
        pos = sum(d > 0 for d in deltas); neg = sum(d < 0 for d in deltas)
        rows[kind] = dict(share=len(sel) / len(d_s), n_state_first=pos, n_output_first=neg, n_tie=len(deltas) - pos - neg)
        print(f"{kind:<22} share {rows[kind]['share']:.3f}  state-first {pos:>5}  output-first {neg:>5}  tie {len(deltas)-pos-neg}")
    both = [(x, y) for x, y in zip(d_s, d_o) if x is not None and y is not None]
    hits = dict(state=np.mean([x is not None for x in d_s]), output=np.mean([y is not None for y in d_o]))
    print("detection rates under the planted null:", {k: round(float(v), 3) for k, v in hits.items()})
    (OUT / "bias.json").write_text(json.dumps(dict(n=len(d_s), by_pair_type=rows,
                                                   detection_rate={k: float(v) for k, v in hits.items()}), indent=2))


PRIMARY_RULE = (
    "Primary fraction: the lowest of 10/25/50% whose planted-null false-positive rate is <= 6% at every "
    "amplitude ratio (0.4, 1, 2.5, measured) and whose power at the measured ratio is >= 80%, at the "
    "smallest seed count where any fraction qualifies. If none qualifies at 1600, the primary is 50% at "
    "1600, declared underpowered."
)


def run_timing_power(n_experiments: int = 100, n_boot: int = 200, sizes=(400, 800, 1600)) -> None:
    """Power of the amended primary test on real old-seed control noise with planted responses.

    Noise: control traces (calibration 10100-10199, confirmatory controls 1000-1099; no switch).
    Shape: the output's empirical mean response on old calibration switch runs (10000-10099), used for
    both signals, so the planted null has identical timing and no bump is built in.
    """
    from regime.analysis import ANALYSIS_CONTROL_SEEDS, CALIBRATION_CONTROL_SEEDS, CALIBRATION_SWITCH_SEEDS
    from regime.early_detection import FRACTIONS, bootstrap_timing

    cal = load_cal("in_context")
    a, W = cal.switch_at, cal.W
    win = slice(a - W, a + W + 1)
    ctl = signal_series(episodes("control", list(CALIBRATION_CONTROL_SEEDS) + list(ANALYSIS_CONTROL_SEEDS), TEST_PAIR), cal)
    sw = signal_series(episodes("switch", CALIBRATION_SWITCH_SEEDS, TEST_PAIR), cal)
    noise = {k: np.stack([x[win] for x in ctl[k]]) for k in ("state_l1", "output")}
    resp = {k: np.stack([x[win] for x in sw[k]]).mean(0) - noise[k].mean(0) for k in noise}
    # Planted runs draw noise from a random time offset in a control trace (post warm-up, where
    # control noise is stationary), so experiments don't share one fixed noise realization.
    # Same trace and offset for both signals keeps their cross-correlation.
    full = {k: np.stack(ctl[k]) for k in noise}
    # Stationary only once the 50-move context is full and warm-up is over.
    first, last = max(cal.t_w, 50) + cal.h, N_ROUNDS - (2 * W + 1)
    sd = {k: float(noise[k].std()) for k in noise}
    shape = np.clip(resp["output"], 0, None)
    shape[:W] = 0
    shape /= shape.max()
    snr = {k: float(resp[k][W:].max() / sd[k]) for k in noise}
    measured_ratio = snr["state_l1"] / snr["output"]
    amp_out = float(resp["output"][W:].max())
    rel = np.arange(-W, W + 1)

    def plant(ratio: float, lead: int, n: int, rng):
        idx = rng.integers(0, len(full["output"]), size=n)
        off = rng.integers(first, last + 1, size=n)
        cols = off[:, None] + np.arange(2 * W + 1)[None, :]
        amp_state = ratio * amp_out * sd["state_l1"] / sd["output"]
        s_shape = np.interp(rel + lead, rel, shape)  # state's response `lead` rounds earlier
        return (full["state_l1"][idx[:, None], cols] + amp_state * s_shape,
                full["output"][idx[:, None], cols] + amp_out * shape)

    rng = np.random.default_rng(9_200_000)
    ratios = {"0.4": 0.4, "1": 1.0, "2.5": 2.5, "measured": measured_ratio}
    table = {}
    for n in sizes:
        row = {}
        for name, r in ratios.items():
            null = [bootstrap_timing(*plant(r, 0, n, rng), W, n_boot=n_boot, seed=int(rng.integers(1e9))) for _ in range(n_experiments)]
            row[f"null_fp_{name}"] = np.mean([np.array(x["p"]) < 0.05 for x in null], 0).round(3).tolist()
            row[f"null_mean_estimate_{name}"] = np.mean([x["estimate"] for x in null], 0).round(2).tolist()
        eff = [bootstrap_timing(*plant(measured_ratio, SESOI, n, rng), W, n_boot=n_boot, seed=int(rng.integers(1e9)))
               for _ in range(n_experiments)]
        row["power_at_measured"] = np.mean([(np.array(x["p"]) < 0.05) & (np.array(x["estimate"]) > 0) for x in eff], 0).round(3).tolist()
        row["mean_estimate_at_effect"] = np.mean([x["estimate"] for x in eff], 0).round(2).tolist()
        table[n] = row
        print(n, json.dumps(row), flush=True)

    primary = None
    for n in sizes:
        for i, f in enumerate(FRACTIONS):
            ok_fp = all(table[n][f"null_fp_{k}"][i] <= 0.06 for k in ratios)
            if ok_fp and table[n]["power_at_measured"][i] >= 0.8:
                primary = dict(fraction=f, n=n)
                break
        if primary:
            break
    if primary is None:
        primary = dict(fraction=0.5, n=max(sizes), underpowered=True)
    (OUT / "timing_power.json").write_text(json.dumps(dict(
        rule=PRIMARY_RULE, fractions=list(FRACTIONS), noise_seeds="controls 10100-10199 and 1000-1099",
        shape_seeds="calibration switch runs 10000-10099 (output mean response)", snr=snr,
        measured_state_to_output_snr_ratio=measured_ratio, experiments=n_experiments, n_boot=n_boot,
        by_n=table, primary=primary), indent=2))
    print("primary:", primary, " measured SNR ratio:", round(measured_ratio, 3), " snr:", snr)


DRY_RUN_SEED_BASE = 39000  # dry runs only, with a non-A/B pair


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def preflight(out: Path) -> str:
    if out.exists():
        raise SystemExit(f"{out} exists: study 2 runs once")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes")
    head = git("rev-parse", "HEAD")
    if TAG not in git("tag", "--points-at", head).split():
        raise SystemExit(f"HEAD is not tagged {TAG}")
    return head


def run_study(dry_run: bool, dry_n: int = 60) -> None:
    """The one confirmatory run of study 2 (amended primary: normalized mean-curve timing)."""
    from regime.early_detection import FRACTIONS, bootstrap_timing, holm, timing_outcome

    power = json.loads((OUT / "timing_power.json").read_text())
    primary = power["primary"]
    if dry_run:
        out, commit = OUT / "dry_run" / "results.json", git("rev-parse", "HEAD") + " (dry run)"
        pair, seeds = DRY_RUN_PAIR, range(DRY_RUN_SEED_BASE, DRY_RUN_SEED_BASE + dry_n)
    else:
        out = OUT / "results.json"
        commit = preflight(out)
        pair, seeds = TEST_PAIR, range(TEST_SWITCH_SEED_BASE, TEST_SWITCH_SEED_BASE + primary["n"])

    cal = load_cal("in_context")
    a, W = cal.switch_at, cal.W
    win = slice(a - W, a + W + 1)
    runs = episodes("switch", seeds, pair)
    # Layer-2 standardization from these runs' own pre-switch rounds (after warm-up): no post-switch information.
    pre = np.concatenate([r["l2"][cal.t_w : a] for r in runs])
    l2_stats = (pre.mean(0), np.where(pre.std(0) > 0, pre.std(0), 1.0))
    sig = signal_series(runs, cal, l2_stats)
    X = {k: np.stack([x[win] for x in v]) for k, v in sig.items()}

    main_res = bootstrap_timing(X["state_l1"], X["output"], W)
    p_adj = holm(main_res["p"])
    by_fraction = {
        f"{f:.0%}": dict(estimate=main_res["estimate"][i], ci95=main_res["ci95"][i], p=main_res["p"][i], p_holm=p_adj[i],
                         outcome=timing_outcome(main_res["estimate"][i], main_res["ci95"][i], p_adj[i]))
        for i, f in enumerate(FRACTIONS)
    }
    i_primary = list(FRACTIONS).index(primary["fraction"])
    control = bootstrap_timing(X["state_l2"], X["output"], W, fractions=(primary["fraction"],))

    results = dict(
        study="early detection (study 2)", commit=commit, tag=TAG, dry_run=dry_run, pair=[q.tolist() for q in pair],
        n_runs=len(runs), seeds=[seeds.start, seeds.stop - 1],
        operational_test="not run: infeasible as designed (power analysis, PREREG_EARLY_DETECTION.md)",
        primary_fraction=f"{primary['fraction']:.0%}", primary_outcome=by_fraction[f"{primary['fraction']:.0%}"]["outcome"],
        convention="estimate = crossing(output) - crossing(state), rounds; positive = state first",
        by_fraction=by_fraction,
        negative_control_layer2_vs_output=dict(fraction=f"{primary['fraction']:.0%}", estimate=control["estimate"][0],
                                               ci95=control["ci95"][0], expected="about 0"),
        underpowered=bool(primary.get("underpowered", False)),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("primary_fraction", "primary_outcome", "by_fraction", "negative_control_layer2_vs_output")}, indent=1))
    print(f"saved {out} (commit {commit})")


def run_strengths(n_boot: int = 2000) -> None:
    """Bootstrap CIs for the response strengths (response over threshold) from the old calibration runs."""
    cal, ctl, sw = old_calibration_signals()
    a, W = cal.switch_at, cal.W
    rng = np.random.default_rng(9_300_000)
    out = {}
    for k in ("state_l1", "output"):
        cm = np.array([run_max(x, a, W) for x in ctl[k]])
        sm = np.array([run_max(x, a, W) for x in sw[k]])
        def strength(c, s_):
            return (np.median(s_) - np.median(c)) / threshold(c, ALPHA_PRIMARY)
        boots = [strength(rng.choice(cm, len(cm)), rng.choice(sm, len(sm))) for _ in range(n_boot)]
        out[k] = dict(estimate=float(strength(cm, sm)), ci95=np.percentile(boots, [2.5, 97.5]).round(3).tolist())
    (OUT / "strengths.json").write_text(json.dumps(dict(seeds="switch 10000-10099, controls 10100-10199", **out), indent=2))
    print(json.dumps(out))


def run_logit_check() -> None:
    """Dry-run pair and seeds only: does using logits as the behavior signal bring the layer-2
    negative control to about 0? (Output scores saturate; the logits don't.)"""
    from exploratory import _pred_from_outputs
    from regime.early_detection import bootstrap_timing

    cal = load_cal("in_context")
    a, W = cal.switch_at, cal.W
    win = slice(a - W, a + W + 1)
    runs = episodes("switch", range(DRY_RUN_SEED_BASE, DRY_RUN_SEED_BASE + 60), DRY_RUN_PAIR)
    pre = np.concatenate([r["l2"][cal.t_w : a] for r in runs])
    sig = signal_series(runs, cal, (pre.mean(0), np.where(pre.std(0) > 0, pre.std(0), 1.0)))
    logits = [np.log(_pred_from_outputs(r["out"])) for r in runs]
    sig["logits"] = [displacement(g - g.mean(1, keepdims=True), cal.h) for g in logits]
    X = {k: np.stack([x[win] for x in v]) for k, v in sig.items()}
    rows = {}
    for s_key in ("state_l2", "state_l1"):
        for b_key in ("output", "logits"):
            r = bootstrap_timing(X[s_key], X[b_key], W, n_boot=2000)
            rows[f"{s_key} vs {b_key}"] = dict(estimate=[round(e, 2) for e in r["estimate"]],
                                               ci95=[[round(c, 2) for c in ci] for ci in r["ci95"]])
            print(f"{s_key} vs {b_key:<7} (10/25/50%): {rows[f'{s_key} vs {b_key}']['estimate']}  CI {rows[f'{s_key} vs {b_key}']['ci95']}")
    (OUT / "dry_run" / "logit_check.json").write_text(json.dumps(dict(
        note="dry-run pair (0.1,0.8,0.1)->(0.1,0.1,0.8), seeds 39000-39059; pipeline validation, not results", **rows), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["power", "power-large", "diagnose", "bias", "timing-power", "strengths", "logit-check", "run"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.which == "run":
        run_study(args.dry_run)
    elif args.which == "power-large":
        run_power(n_experiments=100, sizes=(1600, 3200), n_boot=1000, out_name="power_large.json")
    else:
        {"power": run_power, "diagnose": run_diagnose, "bias": run_bias, "timing-power": run_timing_power,
         "strengths": run_strengths, "logit-check": run_logit_check}[args.which]()


if __name__ == "__main__":
    main()
