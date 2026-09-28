"""Exploratory follow-ups to the confirmatory run (ANALYSIS_PLAN.md,
"Exploratory follow-ups"). Every output is labeled exploratory.

    python scripts/exploratory.py saturation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from regime.analysis import Calibration, lag_estimate, run_test_b, signals
from regime.env import PAYOFF, STRATEGY_A, STRATEGY_B
from regime.runner import EpisodeLog

OUT = Path("results/exploratory")


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(dict(label="exploratory", **obj), indent=2))


# --- 1. planted saturation ----------------------------------------------------

def planted_saturation(beta: float, seed: int, cal: Calibration, d: int = 64, K: int = 50) -> int:
    """True lag is zero: state and behavior are both functions of one belief,
    the move frequency over the last K sampled opponent moves."""
    rng = np.random.default_rng(seed)
    T = cal.n_rounds
    dists = np.stack([STRATEGY_A if t < cal.switch_at else STRATEGY_B for t in range(T)])
    moves = np.array([rng.choice(3, p=p) for p in dists])
    onehot = np.eye(3)[moves]
    csum = np.cumsum(np.vstack([np.zeros(3), onehot]), axis=0)
    counts = csum[1:] - csum[np.maximum(0, np.arange(1, T + 1) - K)]
    belief = (counts + 1) / (counts.sum(1, keepdims=True) + 3)

    M = rng.normal(size=(d, 3))
    states = belief @ M.T + 0.05 * rng.normal(size=(T, d))
    z = beta * np.log(belief)
    pred = np.exp(z - z.max(1, keepdims=True))
    pred /= pred.sum(1, keepdims=True)
    outputs = pred @ PAYOFF.T

    e = np.empty(0)
    log = EpisodeLog(e, moves, e, e, e, e, outputs, states, cal.switch_at, cal.switch_at)
    flat = Calibration(**{**cal.__dict__, "state_mean": [0.0] * d, "state_std": [1.0] * d})
    d_s, d_o = signals(log, flat)
    return lag_estimate(d_s, d_o, cal.switch_at, cal.W, cal.L)


def run_saturation() -> None:
    cal = _load_cal("in_context")
    rows = {}
    for beta in [1, 2, 4, 8]:
        res = run_test_b([planted_saturation(beta, 8_000_000 + s, cal) for s in range(100)])
        rows[str(beta)] = {k: res[k] for k in ("median_lag", "median_ci95", "n_positive", "n_negative", "n_zero", "sign_test_p")}
        print(f"beta={beta}: median lag {res['median_lag']:+.1f}  CI {res['median_ci95']}  "
              f"+/-/0 = {res['n_positive']}/{res['n_negative']}/{res['n_zero']}  sign p={res['sign_test_p']:.3g}", flush=True)
    save("1_planted_saturation", dict(true_lag=0, h=cal.h, W=cal.W, L=cal.L, by_beta=rows))


# --- shared: in-context reruns of the confirmatory episodes --------------------

def _pred_from_outputs(outputs: np.ndarray) -> np.ndarray:
    """Invert outputs = PAYOFF @ p with sum(p) = 1 (exact for RPS)."""
    s0, s1, s2 = outputs.T
    p0 = (1 - s2 + s1) / 3
    return np.clip(np.stack([p0, p0 + s2, p0 - s1], axis=1), 1e-12, 1)


def _icl_run(kind: str, model_kind: str, seed: int) -> dict:
    """One in-context episode; returns the arrays the follow-ups need (not the full log)."""
    import torch

    from regime.analysis import TEST_PAIR, make_agent
    from regime.env import RegimeSwitchOpponent
    from regime.runner import run_episode

    torch.set_num_threads(1)
    cal = _load_cal("in_context")
    if model_kind == "frozen":
        agent = make_agent("in_context")
    else:  # untrained control, follow-up 3
        from regime.agents.in_context import InContextAgent, ModelConfig, MoveTransformer

        torch.manual_seed(0)
        model = MoveTransformer(ModelConfig()).eval()
        agent = InContextAgent(model, temperature=agent_temperature(), move_offsets=_untrained_offsets(model))
    first, second = TEST_PAIR
    opp = (RegimeSwitchOpponent(first, second, switch_at=cal.switch_at) if kind == "switch"
           else RegimeSwitchOpponent.no_switch(first, switch_at=cal.switch_at))
    log = run_episode(agent, opp, cal.n_rounds, seed)
    return dict(states=log.states, outputs=log.outputs, opp_actions=log.opp_actions,
                instant_regret=log.instant_regret, offsets1=agent.move_offsets[1])


_OFFSETS_CACHE = {}


def _untrained_offsets(model):
    from regime.agents.in_context import estimate_move_offsets

    if "u" not in _OFFSETS_CACHE:
        _OFFSETS_CACHE["u"] = estimate_move_offsets(model)
    return _OFFSETS_CACHE["u"]


def agent_temperature() -> float:
    return json.loads(Path("results/tuning/winners.json").read_text())["in_context"]["config"]["temperature"]


def _load_cal(agent: str) -> Calibration:
    from scripts_common import load_cal

    return load_cal(agent)


def icl_runs(kind: str, seeds, model_kind: str = "frozen") -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    with ProcessPoolExecutor(4) as pool:
        return list(pool.map(partial(_icl_run, kind, model_kind), seeds, chunksize=5))


def _disp(x: np.ndarray, h: int) -> np.ndarray:
    from regime.analysis import displacement

    return displacement(x, h)


def _std_state(r: dict, cal: Calibration) -> np.ndarray:
    return (r["states"] - np.array(cal.state_mean)) / np.array(cal.state_std)


def _centered(x: np.ndarray) -> np.ndarray:
    return x - x.mean(1, keepdims=True)


def _lag_summary(lags) -> dict:
    res = run_test_b(lags)
    return {k: res[k] for k in ("median_lag", "median_ci95", "n_positive", "n_negative", "n_zero", "sign_test_p", "wilcoxon_p")}


def _fmt(name: str, r: dict) -> str:
    return (f"{name:<34} median {r['median_lag']:+6.1f}  CI {r['median_ci95']}  "
            f"+/-/0 = {r['n_positive']}/{r['n_negative']}/{r['n_zero']}  sign p={r['sign_test_p']:.3g}")


def run_icl_followups() -> None:
    """Follow-ups 2, 4 and 5 on reruns of the confirmatory in-context episodes."""
    from regime.analysis import ANALYSIS_CONTROL_SEEDS, ANALYSIS_SWITCH_SEEDS

    cal = _load_cal("in_context")
    sw = icl_runs("switch", ANALYSIS_SWITCH_SEEDS)
    ct = icl_runs("control", ANALYSIS_CONTROL_SEEDS)
    a, W, L = cal.switch_at, cal.W, cal.L

    # Reproduction check against the confirmatory lags.
    confirmed = json.loads(Path("results/confirmatory/results.json").read_text())["agents"]["in_context"]["test_b"]["lags"]
    repro = [lag_estimate(_disp(_std_state(r, cal), cal.h), _disp(_centered(r["outputs"]), cal.h), a, W, L) for r in sw]
    assert repro == confirmed, "rerun does not reproduce the confirmatory lags"
    print("reproduction check: rerun lags identical to the confirmatory run", flush=True)

    out = {}
    # 2. behavior = centered logits (log of the predicted distribution), no saturation
    logits = [_centered(np.log(_pred_from_outputs(r["outputs"]))) for r in sw]
    out["2_logit_behavior"] = _lag_summary(
        [lag_estimate(_disp(_std_state(r, cal), cal.h), _disp(g, cal.h), a, W, L) for r, g in zip(sw, logits)])
    print(_fmt("2. behavior = logits", out["2_logit_behavior"]), flush=True)

    # 4. sensitivity to h
    for h in (7, 28):
        key = f"4_h{h}"
        out[key] = _lag_summary(
            [lag_estimate(_disp(_std_state(r, cal), h), _disp(_centered(r["outputs"]), h), a, W, L) for r in sw])
        print(_fmt(f"4. h = {h}", out[key]), flush=True)

    # 5. onset comparison (positive = state first), thresholds from controls over the window
    win = slice(a - W, a + W + 1)
    sig = {
        "state": lambda r: _disp(_std_state(r, cal), cal.h),
        "output": lambda r: _disp(_centered(r["outputs"]), cal.h),
    }
    thr = {k: np.nanpercentile(np.concatenate([f(r)[win] for r in ct]), 95) for k, f in sig.items()}

    def onset(x, t):
        idx = np.flatnonzero(x[a : a + W + 1] > t)
        return int(idx[0]) if len(idx) else None

    diffs, missing = [], 0
    for r in sw:
        o_s, o_o = onset(sig["state"](r), thr["state"]), onset(sig["output"](r), thr["output"])
        if o_s is None or o_o is None:
            missing += 1
        else:
            diffs.append(o_o - o_s)
    out["5_onset"] = dict(_lag_summary(diffs), n_missing_onset=missing,
                          thresholds={k: float(v) for k, v in thr.items()})
    print(_fmt("5. onset(behavior) - onset(state)", out["5_onset"]) + f"  missing {missing}", flush=True)
    save("2_4_5_in_context", out)


def run_untrained() -> None:
    """Follow-up 3: Test B pipeline on an untrained network, corrected and raw state."""
    from regime.analysis import ANALYSIS_SWITCH_SEEDS, CALIBRATION_CONTROL_SEEDS

    cal = _load_cal("in_context")
    a, W, L, h = cal.switch_at, cal.W, cal.L, cal.h
    ct = icl_runs("control", CALIBRATION_CONTROL_SEEDS, "untrained")
    base = np.concatenate([r["states"][cal.t_w:] for r in ct])
    mean, std = base.mean(0), base.std(0)
    std = np.where(std > 0, std, 1.0)
    sw = icl_runs("switch", ANALYSIS_SWITCH_SEEDS, "untrained")

    out = {}
    corrected = [lag_estimate(_disp((r["states"] - mean) / std, h), _disp(_centered(r["outputs"]), h), a, W, L) for r in sw]
    raw = [lag_estimate(_disp(r["states"] + r["offsets1"][r["opp_actions"]], h), _disp(_centered(r["outputs"]), h), a, W, L)
           for r in sw]
    out["3_untrained_corrected"] = _lag_summary(corrected)
    out["3_untrained_raw"] = _lag_summary(raw)
    print(_fmt("3. untrained, corrected state", out["3_untrained_corrected"]), flush=True)
    print(_fmt("3. untrained, raw state", out["3_untrained_raw"]), flush=True)
    save("3_untrained", out)


HOLDOUT_CONTROL_SEED_BASE = 7_000_000


def holdout_control_pair(i: int):
    """Follow-up 8: an allowed hard-switch pair about as far apart as A and B."""
    from regime.pretrain import _sample_strategy, tv

    rng = np.random.default_rng(HOLDOUT_CONTROL_SEED_BASE + i)
    p = _sample_strategy(rng)
    while True:
        q = _sample_strategy(rng)
        if 0.35 <= tv(p, q) <= 0.45:
            return p, q


def _regret_run(agent_name: str, which: str, i: int) -> dict:
    """Follow-ups 6 and 8: one 200/600 hard-switch episode, regret summaries only."""
    import torch

    from regime.analysis import TEST_PAIR, make_agent
    from regime.env import RegimeSwitchOpponent
    from regime.metrics import excess_regret
    from regime.runner import run_episode

    torch.set_num_threads(1)
    if which == "test":
        (p, q), seed = TEST_PAIR, i
    else:
        (p, q), seed = holdout_control_pair(i), HOLDOUT_CONTROL_SEED_BASE + i
    log = run_episode(make_agent(agent_name), RegimeSwitchOpponent(p, q, switch_at=200), 600, seed)
    return dict(total=float(log.instant_regret.sum()), excess=excess_regret(log))


def _weight_lag(seed: int) -> int:
    """Follow-up 7: fine-tuning agent, state = flattened weights (unstandardized)."""
    from regime.agents.online import FineTuneAgent
    from regime.analysis import TEST_PAIR
    from regime.env import RegimeSwitchOpponent
    from regime.runner import run_episode

    cal = _load_cal("fine_tune")
    cfg = json.loads(Path("results/tuning/winners.json").read_text())["fine_tune"]["config"]

    class WeightState(FineTuneAgent):
        def internal_state(self):
            return self.weight_vector()

    first, second = TEST_PAIR
    log = run_episode(WeightState(**cfg), RegimeSwitchOpponent(first, second, switch_at=cal.switch_at), cal.n_rounds, seed)
    return lag_estimate(_disp(log.states, cal.h), _disp(_centered(log.outputs), cal.h), cal.switch_at, cal.W, cal.L)


def run_remaining() -> None:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from regime.analysis import AGENTS, ANALYSIS_SWITCH_SEEDS

    out = {}
    with ProcessPoolExecutor(4) as pool:
        for agent in AGENTS:
            test = list(pool.map(partial(_regret_run, agent, "test"), ANALYSIS_SWITCH_SEEDS, chunksize=5))
            ctrl = list(pool.map(partial(_regret_run, agent, "holdout_control"), range(100), chunksize=5))

            def ms(rows, k):
                x = np.array([r[k] for r in rows])
                return [float(x.mean()), float(x.std() / np.sqrt(len(x)))]

            out[agent] = {"6_total_regret_A_to_B": ms(test, "total"), "excess_regret_A_to_B": ms(test, "excess"),
                          "8_excess_regret_holdout_control": ms(ctrl, "excess"),
                          "8_total_regret_holdout_control": ms(ctrl, "total")}
            o = out[agent]
            print(f"{agent:<13} total regret A->B {o['6_total_regret_A_to_B'][0]:6.1f} ± {o['6_total_regret_A_to_B'][1]:.1f}   "
                  f"excess A->B {o['excess_regret_A_to_B'][0]:5.1f} ± {o['excess_regret_A_to_B'][1]:.1f}   "
                  f"excess control pairs {o['8_excess_regret_holdout_control'][0]:5.1f} ± {o['8_excess_regret_holdout_control'][1]:.1f}",
                  flush=True)
        lags = list(pool.map(_weight_lag, ANALYSIS_SWITCH_SEEDS, chunksize=5))
    out["7_fine_tune_weight_displacement"] = _lag_summary(lags)
    print(_fmt("7. fine-tune, weight state", out["7_fine_tune_weight_displacement"]), flush=True)
    save("6_7_8", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["saturation", "icl", "untrained", "remaining"])
    args = ap.parse_args()
    {"saturation": run_saturation, "icl": run_icl_followups, "untrained": run_untrained,
     "remaining": run_remaining}[args.which]()


if __name__ == "__main__":
    main()
