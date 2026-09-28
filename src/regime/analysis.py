"""Calibration and confirmatory analysis (ANALYSIS_PLAN.md and its amendments).

Calibration derives per-agent parameters (T_w, h, W, L, switch_at,
n_rounds, state standardization) from seeds 10000+. It never computes
Tests A or B. The confirmatory analysis runs Tests A and B on seeds 0-99.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

from regime.env import STRATEGY_A, STRATEGY_B, RegimeSwitchOpponent
from regime.metrics import moving_average, recovery_time
from regime.runner import EpisodeLog, run_episode

CALIBRATION_SWITCH_SEEDS = range(10000, 10100)
CALIBRATION_CONTROL_SEEDS = range(10100, 10200)
ANALYSIS_SWITCH_SEEDS = range(0, 100)
ANALYSIS_CONTROL_SEEDS = range(1000, 1100)
DEFAULT_SWITCH_AT, DEFAULT_ROUNDS = 200, 600
AGENTS = ("in_context", "fine_tune", "change_aware")


# --- agents -----------------------------------------------------------------

def make_agent(name: str, winners_path: str | Path = "results/tuning/winners.json"):
    w = json.loads(Path(winners_path).read_text())[name]
    if name == "in_context":
        from regime.agents.in_context import InContextAgent, load_frozen, weights_digest

        model, offsets, _ = load_frozen(w["model"])
        if weights_digest(model) != w["weights_sha256"]:
            raise ValueError("frozen model does not match winners.json")
        return InContextAgent(model, temperature=w["config"]["temperature"], move_offsets=offsets)
    from regime.agents.online import ChangeAwareAgent, FineTuneAgent

    return {"fine_tune": FineTuneAgent, "change_aware": ChangeAwareAgent}[name](**w["config"])


def _run(name: str, kind: str, switch_at: int, n_rounds: int, seed: int) -> EpisodeLog:
    import torch

    torch.set_num_threads(1)
    opp = (RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B, switch_at=switch_at) if kind == "switch"
           else RegimeSwitchOpponent.no_switch(STRATEGY_A, switch_at=switch_at))
    return run_episode(make_agent(name), opp, n_rounds, seed)


def run_many(name: str, kind: str, seeds, switch_at: int, n_rounds: int, workers: int = 4) -> list[EpisodeLog]:
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(partial(_run, name, kind, switch_at, n_rounds), seeds, chunksize=5))


# --- calibration ------------------------------------------------------------

@dataclass
class Calibration:
    agent: str
    t_w: int
    median_recovery: float | None
    p90_recovery: float | None
    n_censored: int
    runnable: bool
    h: int | None
    W: int | None
    L: int | None
    switch_at: int | None
    n_rounds: int | None
    state_mean: list[float]
    state_std: list[float]


def censored_quantile(times: list[int | None], q: float) -> float | None:
    """Quantile with censored runs (None) ranked above every observed time.
    Returns None if the quantile falls on a censored run."""
    n = len(times)
    observed = sorted(t for t in times if t is not None)
    k = int(np.ceil(q * n)) - 1  # 0-based order statistic (nearest-rank)
    return float(observed[k]) if k < len(observed) else None


def settle_round(mean_regret: np.ndarray, tail: int = 100, frac: float = 0.1) -> int:
    """First round after which m(t) stays within frac * |m(0) - m_final| of m_final."""
    final = mean_regret[-tail:].mean()
    band = frac * abs(mean_regret[0] - final)
    outside = np.flatnonzero(np.abs(mean_regret - final) > band)
    return 0 if len(outside) == 0 else int(outside[-1] + 1)


def safe_recovery(log: EpisodeLog) -> int | None:
    try:
        return recovery_time(log)
    except ValueError:  # no pre-switch edge: censored (ANALYSIS_PLAN.md amendment)
        return None


def calibrate(name: str, switch_logs: list[EpisodeLog], control_logs: list[EpisodeLog]) -> Calibration:
    times = [safe_recovery(log) for log in switch_logs]
    med, p90 = censored_quantile(times, 0.5), censored_quantile(times, 0.9)
    m = np.mean([moving_average(log.instant_regret, 20) for log in control_logs], axis=0)
    t_w = settle_round(m)

    states = np.concatenate([log.states[t_w:DEFAULT_SWITCH_AT] for log in control_logs])
    mean, std = states.mean(0), states.std(0)
    std = np.where(std > 0, std, 1.0)

    runnable = med is not None and p90 is not None
    h = W = L = switch_at = n_rounds = None
    if runnable:
        h = max(5, int(round(med / 2)))
        W = max(int(2 * p90), 3 * h)
        L = W // 2
        switch_at = max(DEFAULT_SWITCH_AT, t_w + h + W)
        n_rounds = max(DEFAULT_ROUNDS, switch_at + W + 1)
    return Calibration(name, t_w, med, p90, times.count(None), runnable, h, W, L, switch_at, n_rounds,
                       mean.tolist(), std.tolist())


# --- signals ----------------------------------------------------------------

def displacement(x: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[h:] = np.linalg.norm(x[h:] - x[:-h], axis=1)
    return out


def signals(log: EpisodeLog, cal: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """State and behavior displacement series (D_s, D_o) for one run."""
    states = (log.states - np.array(cal.state_mean)) / np.array(cal.state_std)
    outputs = log.outputs - log.outputs.mean(1, keepdims=True)
    return displacement(states, cal.h), displacement(outputs, cal.h)


# --- Test A -----------------------------------------------------------------

def peak_state_displacement(d_s: np.ndarray, switch_at: int, W: int) -> float:
    return float(np.nanmax(d_s[switch_at : switch_at + W + 1]))


def run_test_a(switch_stats: list[float], control_stats: list[float]) -> dict:
    from scipy.stats import mannwhitneyu

    res = mannwhitneyu(switch_stats, control_stats, alternative="greater")
    return dict(U=float(res.statistic), p=float(res.pvalue), passed=bool(res.pvalue < 0.05),
                n_switch=len(switch_stats), n_control=len(control_stats))


# --- Test B -----------------------------------------------------------------

def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def lag_estimate(d_s: np.ndarray, d_o: np.ndarray, switch_at: int, W: int, L: int) -> int:
    """argmax_k corr(D_s(t), D_o(t + k)) within the analysis window, k in [-L, L].

    Positive k: the state's displacement rises k rounds before the behavior's.
    Linear (non-circular): at each k only overlapping rounds are used. Ties go
    to the smallest |k|; a tie between +k and -k is 0.
    """
    s = _z(d_s[switch_at - W : switch_at + W + 1])
    o = _z(d_o[switch_at - W : switch_at + W + 1])
    n = len(s)
    corr = {}
    for k in range(-L, L + 1):
        a, b = (s[: n - k], o[k:]) if k >= 0 else (s[-k:], o[: n + k])
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a @ a) * (b @ b))
        corr[k] = float(a @ b / den) if den > 0 else -np.inf
    best = max(corr.values())
    ties = [k for k, c in corr.items() if np.isclose(c, best, rtol=0, atol=1e-12)]
    m = min(abs(k) for k in ties)
    closest = {k for k in ties if abs(k) == m}
    return 0 if len(closest) > 1 else closest.pop()


def run_test_b(lags: list[int], n_boot: int = 10_000, seed: int = 0) -> dict:
    from scipy.stats import binomtest, wilcoxon

    lags = np.asarray(lags)
    nz = lags[lags != 0]
    n_pos = int((nz > 0).sum())
    sign_p = float(binomtest(n_pos, len(nz)).pvalue) if len(nz) else 1.0
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(lags, size=(n_boot, len(lags))), axis=1)
    wil = float(wilcoxon(nz).pvalue) if len(nz) > 0 and np.any(nz != nz[0]) else None
    return dict(
        n=len(lags), n_zero=int((lags == 0).sum()), n_positive=n_pos, n_negative=int((nz < 0).sum()),
        sign_test_p=sign_p, significant=bool(sign_p < 0.05), median_lag=float(np.median(lags)),
        median_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        wilcoxon_p=wil, lags=lags.tolist(),
    )


def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj, indent=2))
