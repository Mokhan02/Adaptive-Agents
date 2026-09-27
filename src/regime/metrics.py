from __future__ import annotations

import numpy as np

from regime.runner import EpisodeLog


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; entry t averages x[max(0, t-window+1) : t+1]."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    return (c[idx] - c[lo]) / (idx - lo)


def cumulative_regret(log: EpisodeLog) -> np.ndarray:
    return np.cumsum(log.instant_regret)


def windowed_regret(log: EpisodeLog, window: int = 20) -> np.ndarray:
    return moving_average(log.instant_regret, window)


def recovery_time(
    log: EpisodeLog,
    window: int = 20,
    threshold: float = 0.9,
    sustain: int = 20,
    baseline_rounds: int = 50,
) -> int | None:
    """Rounds after the switch completes until performance recovers.

    Performance is the trailing `window`-round mean of expected reward, taken
    as a fraction of the oracle's (so regimes with different achievable reward
    are comparable). The agent has recovered once it holds at least
    `threshold` of its pre-switch fraction-of-oracle for `sustain` consecutive
    rounds. Returns None if it never recovers.

    Raises ValueError if the agent had no edge before the switch, since
    "recovering" to zero performance is trivially satisfied. For gradual
    switches adaptive agents often recover during the transition and score 0
    here; use `excess_regret` to compare them.
    """
    perf = moving_average(log.expected_rewards, window)
    oracle = moving_average(log.oracle_rewards, window)
    pre = slice(max(0, log.switch_at - baseline_rounds), log.switch_at)
    pre_eff = perf[pre].mean() / oracle[pre].mean()
    if pre_eff <= 0:
        raise ValueError("agent had no pre-switch edge; recovery time is undefined")
    target = threshold * pre_eff * oracle

    ok = perf >= target
    run = 0
    for t in range(log.switch_end, len(ok)):
        run = run + 1 if ok[t] else 0
        if run == sustain:
            return t - sustain + 1 - log.switch_end
    return None


def excess_regret(log: EpisodeLog, horizon: int = 100, baseline_rounds: int = 50) -> float:
    """Regret caused by the switch: total regret from the start of the switch
    through `horizon` rounds after it completes, minus what the agent's
    pre-switch regret rate would have accumulated over the same span.

    Regret is taken as a fraction of the oracle's value each round, since a
    gradual switch passes through mixtures that are less exploitable; raw
    regret would shrink there for every agent, adaptive or not."""
    r = log.instant_regret / np.where(log.oracle_rewards > 0, log.oracle_rewards, np.nan)
    r = np.nan_to_num(r)
    span = slice(log.switch_at, min(len(r), log.switch_end + horizon))
    pre_rate = r[max(0, log.switch_at - baseline_rounds) : log.switch_at].mean()
    return float(r[span].sum() - pre_rate * (span.stop - span.start))


def representational_drift(states: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Distance between consecutive internal states; entry t compares t with t-1.
    Entry 0 is NaN."""
    prev, cur = states[:-1], states[1:]
    if metric == "cosine":
        num = np.sum(prev * cur, axis=1)
        den = np.linalg.norm(prev, axis=1) * np.linalg.norm(cur, axis=1)
        d = 1.0 - num / np.where(den == 0, 1.0, den)
    elif metric == "l2":
        d = np.linalg.norm(cur - prev, axis=1)
    else:
        raise ValueError(f"unknown metric: {metric}")
    return np.concatenate([[np.nan], d])
