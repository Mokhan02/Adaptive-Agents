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


def _two_layer_run(kind: str, seed: int) -> dict:
    """Study 1's episode (600 rounds, A -> B at 200) logging corrected layer-1 and layer-2 residuals."""
    import torch

    from regime.agents.in_context import InContextAgent
    from regime.analysis import TEST_PAIR, make_agent
    from regime.env import RegimeSwitchOpponent
    from regime.runner import run_episode

    torch.set_num_threads(1)
    cal = _load_cal("in_context")
    base = make_agent("in_context")

    class TwoLayers(InContextAgent):
        def internal_state(self):
            r = self.residual_streams()
            return np.concatenate([r[1], r[2]])

    agent = TwoLayers(base.model, temperature=base.temperature, move_offsets=base.move_offsets)
    first, second = TEST_PAIR
    opp = (RegimeSwitchOpponent(first, second, switch_at=cal.switch_at) if kind == "switch"
           else RegimeSwitchOpponent.no_switch(first, switch_at=cal.switch_at))
    log = run_episode(agent, opp, cal.n_rounds, seed)
    d = log.states.shape[1] // 2
    return dict(l1=log.states[:, :d], l2=log.states[:, d:], out=log.outputs, opp=log.opp_actions,
                offsets2=base.move_offsets[2])


def run_structural_null() -> None:
    """ANALYSIS_PLAN.md, "Structural-null check": layer 2 vs output with study 1's exact pipeline."""
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from regime.analysis import ANALYSIS_SWITCH_SEEDS, CALIBRATION_CONTROL_SEEDS

    cal = _load_cal("in_context")
    a, W, L, h = cal.switch_at, cal.W, cal.L, cal.h
    with ProcessPoolExecutor(4) as pool:
        sw = list(pool.map(partial(_two_layer_run, "switch"), ANALYSIS_SWITCH_SEEDS, chunksize=5))
        ct = list(pool.map(partial(_two_layer_run, "control"), CALIBRATION_CONTROL_SEEDS, chunksize=5))

    out_d = [_disp(_centered(r["out"]), h) for r in sw]
    l1 = [lag_estimate(_disp(_std_state({"states": r["l1"]}, cal), h), o, a, W, L) for r, o in zip(sw, out_d)]
    confirmed = json.loads(Path("results/confirmatory/results.json").read_text())["agents"]["in_context"]["test_b"]["lags"]
    assert l1 == confirmed, "rerun does not reproduce study 1's layer-1 lags"
    print("reproduction check: layer-1 lags identical to study 1", flush=True)

    base = np.concatenate([r["l2"][cal.t_w :] for r in ct])
    mean2, std2 = base.mean(0), np.where(base.std(0) > 0, base.std(0), 1.0)
    corrected = [lag_estimate(_disp((r["l2"] - mean2) / std2, h), o, a, W, L) for r, o in zip(sw, out_d)]
    raw = [lag_estimate(_disp(r["l2"] + r["offsets2"][r["opp"]], h), o, a, W, L) for r, o in zip(sw, out_d)]

    res = {"layer1_corrected_reference": _lag_summary(l1), "layer2_corrected": _lag_summary(corrected),
           "layer2_raw": _lag_summary(raw)}
    for k, v in res.items():
        print(_fmt(k, v), flush=True)

    c = res["layer2_corrected"]
    lo, hi = c["median_ci95"]
    n_nz = c["n_positive"] + c["n_negative"]
    frac_neg = c["n_negative"] / n_nz if n_nz else 0.0
    overlaps = lo <= -2 and hi >= -3
    if -3.5 <= c["median_lag"] <= -1.5 and overlaps and frac_neg >= 0.8:
        verdict = "comparable: artifact explanation supported; reframe study 1's headline"
    elif (c["median_lag"] > -1.0 or frac_neg < 0.6) and not overlaps:
        verdict = "meaningfully different: artifact explanation weakened; layers need their own account"
    else:
        verdict = "intermediate: partly explained by curve shape"
    res["share_negative_among_nonzero"] = frac_neg
    res["verdict"] = verdict
    print("verdict:", verdict)
    lags = dict(layer1_corrected=l1, layer2_corrected=corrected, layer2_raw=raw)
    path = OUT / "structural_null_layer2.json"
    if path.exists():  # rerun for the per-seed lags: must reproduce the committed summaries
        committed = json.loads(path.read_text())
        assert all(committed[k] == res[k] for k in ("layer1_corrected_reference", "layer2_corrected", "layer2_raw")), \
            "rerun does not reproduce the committed structural-null summaries"
        print("rerun reproduces the committed summaries")
    else:
        save("structural_null_layer2", res)
    save("structural_null_layer2_lags", lags)


def _l1_run(kind: str, seed: int) -> dict:
    """Frozen in-context agent; corrected layer-1 state and output scores. kind: 'switch' (A -> B,
    study 1's structure) or 'pretrain' (a pretraining-distribution opponent from `seed`)."""
    import torch

    from regime.analysis import TEST_PAIR, make_agent
    from regime.env import RegimeSwitchOpponent
    from regime.pretrain import sample_pretraining_opponent
    from regime.runner import run_episode

    torch.set_num_threads(1)
    cal = _load_cal("in_context")
    if kind == "switch":
        opp = RegimeSwitchOpponent(*TEST_PAIR, switch_at=cal.switch_at)
    else:
        opp = sample_pretraining_opponent(np.random.default_rng(seed), cal.n_rounds)
    log = run_episode(make_agent("in_context"), opp, cal.n_rounds, seed)
    return dict(states=log.states, out=log.outputs)


def _runs(kind: str, seeds) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    with ProcessPoolExecutor(4) as pool:
        return list(pool.map(partial(_l1_run, kind), seeds, chunksize=5))


def _logits(out: np.ndarray) -> np.ndarray:
    return _centered(np.log(_pred_from_outputs(out)))


def run_dimension() -> None:
    """ANALYSIS_PLAN.md, "Dimension diagnostics": D1 readout projection, D2 random-projection sweep."""
    from scipy.stats import spearmanr

    from regime.analysis import ANALYSIS_SWITCH_SEEDS, CALIBRATION_SWITCH_SEEDS

    cal = _load_cal("in_context")
    a, W, L, h = cal.switch_at, cal.W, cal.L, cal.h
    std = lambda r: _std_state(r, cal)

    # --- D1: fit the readout on pretraining-distribution episodes
    fit = _runs("pretrain", range(4_100_000, 4_100_100))
    X = np.concatenate([std(r)[64:] for r in fit])
    Y = np.concatenate([_logits(r["out"])[64:] for r in fit])
    lam = 1e-3 * np.trace(X.T @ X) / X.shape[1]
    B = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)  # (64, 3)
    U, S, _ = np.linalg.svd(B, full_matrices=False)
    Q = U[:, :2]  # rank 2: the centered logits have 2 degrees of freedom

    def r2(runs):
        Xh = np.concatenate([std(r)[64:] for r in runs])
        Yh = np.concatenate([_logits(r["out"])[64:] for r in runs])
        return float(1 - ((Yh - Xh @ B) ** 2).sum() / ((Yh - Yh.mean(0)) ** 2).sum())

    r2_pre = r2(_runs("pretrain", range(4_200_000, 4_200_050)))
    r2_ab = r2(_runs("switch", CALIBRATION_SWITCH_SEEDS))
    valid = r2_pre >= 0.70 and r2_ab >= 0.70
    print(f"D1 validity: held-out R^2 pretraining {r2_pre:.3f}, A->B calibration {r2_ab:.3f} -> {'valid' if valid else 'INVALID'}",
          flush=True)

    sw = _runs("switch", ANALYSIS_SWITCH_SEEDS)
    out_d = [_disp(_centered(r["out"]), h) for r in sw]
    full = [lag_estimate(_disp(std(r), h), o, a, W, L) for r, o in zip(sw, out_d)]
    confirmed = json.loads(Path("results/confirmatory/results.json").read_text())["agents"]["in_context"]["test_b"]["lags"]
    assert full == confirmed, "rerun does not reproduce study 1's layer-1 lags"
    print("reproduction check: layer-1 lags identical to study 1", flush=True)

    def lags_for(P):
        return [lag_estimate(_disp(std(r) @ P, h), o, a, W, L) for r, o in zip(sw, out_d)]

    d1 = _lag_summary(lags_for(Q))
    lo, hi = d1["median_ci95"]
    med = d1["median_lag"]
    if not valid:
        d1_verdict = "invalid: readout subspace does not carry the output's information on this data"
    elif med > 0 and lo > 0:
        d1_verdict = "reverses: readout moves before the output"
    elif med > -1.0 and hi >= 0:
        d1_verdict = "shrinks: mismatch explanation supported"
    elif med <= -2.0 and hi < 0:
        d1_verdict = "persists"
    else:
        d1_verdict = "partial"
    print(_fmt("D1 readout (k=2)", d1) + f"  -> {d1_verdict}", flush=True)

    # --- D2: random orthonormal projections
    rng = np.random.default_rng(6_100_000)
    ks, sweep = (2, 4, 8, 16, 32, 64), {}
    pairs = []
    for k in ks:
        meds = []
        for _ in range(20):
            P, _r = np.linalg.qr(rng.normal(size=(64, k)))
            lags = lags_for(P)
            if k == 64:
                assert lags == confirmed, "a rotation changed the lags"
            meds.append(float(np.median(lags)))
            pairs.append((k, abs(meds[-1])))
        sweep[k] = dict(median_lags=meds, mean_abs=float(np.mean(np.abs(meds))),
                        range=[float(min(meds)), float(max(meds))])
        print(f"D2 k={k:>2}: mean |median lag| {sweep[k]['mean_abs']:.2f}  range {sweep[k]['range']}", flush=True)
    rho, p_two = spearmanr([k for k, _ in pairs], [v for _, v in pairs])
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    if p_one < 0.05 and sweep[2]["mean_abs"] <= 1.0:
        d2_verdict = "shrinks toward 0: mismatch supported"
    elif p_one < 0.05:
        d2_verdict = "partial dose-response"
    else:
        d2_verdict = "flat: not dimension per se"
    readout_vs_random = float(np.mean(np.array(sweep[2]["median_lags"]) <= med))
    print(f"D2 Spearman rho {rho:.3f}, one-sided p {p_one:.3g} -> {d2_verdict}; "
          f"share of random k=2 medians <= readout's: {readout_vs_random:.2f}", flush=True)
    save("dimension_diagnostics", dict(
        d1=dict(r2_heldout_pretraining=r2_pre, r2_calibration_A_to_B=r2_ab, valid=valid, ridge_lambda=float(lam),
                singular_values=S.tolist(), test_b=d1, verdict=d1_verdict),
        d2=dict(by_k={str(k): v for k, v in sweep.items()}, spearman_rho=float(rho), spearman_p_one_sided=float(p_one),
                verdict=d2_verdict, share_random_k2_at_or_below_readout=readout_vs_random)))


def _layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-round LayerNorm without the affine part (rows = rounds)."""
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(-1, keepdims=True) + eps)


def _ln_offsets(n_episodes: int = 200) -> np.ndarray:
    """Per-move mean of the *normalized* final-position residual, per layer (the pre-registered
    move-offset procedure, in LayerNorm space). Returns (n_layers + 1, 3, d)."""
    import torch

    from regime.analysis import make_agent
    from regime.agents.in_context import MOVE_OFFSET_SEED_BASE
    from regime.pretrain import sample_moves, sample_pretraining_opponent

    model = make_agent("in_context").model
    K = model.cfg.context
    sums = np.zeros((model.cfg.n_layers + 1, 3, model.cfg.d_model))
    counts = np.zeros(3)
    with torch.no_grad():
        for i in range(n_episodes):
            rng = np.random.default_rng(MOVE_OFFSET_SEED_BASE + i)
            moves = sample_moves(sample_pretraining_opponent(rng, 600).distributions(600), rng)
            windows = torch.as_tensor(np.lib.stride_tricks.sliding_window_view(moves, K).copy())
            _, resid = model(windows)
            last = moves[K - 1 :]
            for layer, r in enumerate(resid):
                np.add.at(sums[layer], last, _layernorm(r[:, -1].double().numpy()))
            counts += np.bincount(last, minlength=3)
    return sums / counts[None, :, None]


def _raw_run(kind: str, seed: int) -> dict:
    """Study 1's episode, logging the *raw* (uncorrected) layer-1 and layer-2 residuals."""
    import torch

    from regime.agents.in_context import InContextAgent
    from regime.analysis import TEST_PAIR, make_agent
    from regime.env import RegimeSwitchOpponent
    from regime.runner import run_episode

    torch.set_num_threads(1)
    cal = _load_cal("in_context")
    base = make_agent("in_context")

    class Raw(InContextAgent):
        def internal_state(self):
            r = self.residual_streams(corrected=False)
            return np.concatenate([r[1], r[2]])

    agent = Raw(base.model, temperature=base.temperature, move_offsets=base.move_offsets)
    opp = (RegimeSwitchOpponent(*TEST_PAIR, switch_at=cal.switch_at) if kind == "switch"
           else RegimeSwitchOpponent.no_switch(TEST_PAIR[0], switch_at=cal.switch_at))
    log = run_episode(agent, opp, cal.n_rounds, seed)
    d = log.states.shape[1] // 2
    return dict(raw1=log.states[:, :d], raw2=log.states[:, d:], out=log.outputs, opp=log.opp_actions,
                offsets=base.move_offsets)


def run_layernorm() -> None:
    """ANALYSIS_PLAN.md, "LayerNorm diagnostic"."""
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from regime.analysis import ANALYSIS_SWITCH_SEEDS, CALIBRATION_CONTROL_SEEDS

    cal = _load_cal("in_context")
    a, W, L, h = cal.switch_at, cal.W, cal.L, cal.h
    with ProcessPoolExecutor(4) as pool:
        sw = list(pool.map(partial(_raw_run, "switch"), ANALYSIS_SWITCH_SEEDS, chunksize=5))
        ct = list(pool.map(partial(_raw_run, "control"), CALIBRATION_CONTROL_SEEDS, chunksize=5))
    out_d = [_disp(_centered(r["out"]), h) for r in sw]

    # Reproduction: unnormalized corrected layer 1 through study 1's pipeline.
    l1 = [lag_estimate(_disp(_std_state({"states": r["raw1"] - r["offsets"][1, r["opp"]]}, cal), h), o, a, W, L)
          for r, o in zip(sw, out_d)]
    confirmed = json.loads(Path("results/confirmatory/results.json").read_text())["agents"]["in_context"]["test_b"]["lags"]
    assert l1 == confirmed, "rerun does not reproduce study 1's layer-1 lags"
    print("reproduction check: layer-1 lags identical to study 1", flush=True)

    ln_off = _ln_offsets()
    res = {}
    for layer, key in ((1, "raw1"), (2, "raw2")):
        corr = lambda r: _layernorm(r[key]) - ln_off[layer][r["opp"]]
        base = np.concatenate([corr(r)[cal.t_w :] for r in ct])
        mean, std = base.mean(0), np.where(base.std(0) > 0, base.std(0), 1.0)
        lags = [lag_estimate(_disp((corr(r) - mean) / std, h), o, a, W, L) for r, o in zip(sw, out_d)]
        res[f"layer{layer}_layernorm"] = _lag_summary(lags)
        print(_fmt(f"layer {layer}, LayerNorm-normalized", res[f"layer{layer}_layernorm"]), flush=True)

    r1 = res["layer1_layernorm"]
    lo, hi = r1["median_ci95"]
    m = r1["median_lag"]
    if -1.0 <= m <= 1.0 and lo <= 0 <= hi:
        verdict = "supports magnitude drift through LayerNorm (one plausible mechanism)"
    elif m <= -1.5 and hi < 0:
        verdict = "disconfirmed: mechanism unresolved"
    else:
        verdict = "ambiguous: mechanism unresolved"
    res["verdict"] = verdict
    print("verdict:", verdict)
    save("layernorm_diagnostic", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["saturation", "icl", "untrained", "remaining", "structural-null", "dimension", "layernorm"])
    args = ap.parse_args()
    {"saturation": run_saturation, "icl": run_icl_followups, "untrained": run_untrained,
     "remaining": run_remaining, "structural-null": run_structural_null, "dimension": run_dimension,
     "layernorm": run_layernorm}[args.which]()


if __name__ == "__main__":
    main()
