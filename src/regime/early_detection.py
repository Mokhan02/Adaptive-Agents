"""Study 2: does the in-context agent's state detect a switch before its
output does, at a matched per-run false-alarm rate? (PREREG_EARLY_DETECTION.md)

Detector, identical for every signal: the first round in the detection window
[switch_at, switch_at + W] at which the signal's displacement exceeds its
threshold. The threshold is the (1 - alpha) quantile of the signal's per-run
maximum displacement over that window in no-switch control runs, so each
signal false-alarms in a fraction alpha of control runs.
"""

from __future__ import annotations

import numpy as np

ALPHA_PRIMARY, ALPHA_SECONDARY = 0.05, 0.01
SESOI = 2  # smallest effect of interest: the state detects >= 2 rounds earlier

THRESHOLD_CONTROL_SEEDS = range(20000, 21000)
HELDOUT_CONTROL_SEEDS = range(21000, 22000)
TEST_SWITCH_SEED_BASE = 30000  # test switch runs: 30000 .. 30000 + N - 1


def run_max(d: np.ndarray, switch_at: int, W: int) -> float:
    return float(np.nanmax(d[switch_at : switch_at + W + 1]))


def threshold(control_maxes, alpha: float) -> float:
    """(1 - alpha) quantile of per-run maxima ('higher' method: never below
    an observed maximum, so the false-alarm rate is at most alpha in-sample)."""
    return float(np.quantile(np.asarray(control_maxes), 1 - alpha, method="higher"))


def false_alarm_rate(control_maxes, tau: float) -> float:
    return float(np.mean(np.asarray(control_maxes) > tau))


def binomial_interval(n: int, p: float, level: float = 0.99) -> tuple[float, float]:
    """Central interval for an empirical rate when the true rate is p."""
    from scipy.stats import binom

    lo, hi = binom.ppf([(1 - level) / 2, 1 - (1 - level) / 2], n, p)
    return float(lo / n), float(hi / n)


def detection_delay(d: np.ndarray, tau: float, switch_at: int, W: int) -> int | None:
    """Rounds after the switch until d first exceeds tau; None if it never does."""
    hits = np.flatnonzero(d[switch_at : switch_at + W + 1] > tau)
    return int(hits[0]) if len(hits) else None


def paired_differences(delays_state, delays_output, W: int) -> tuple[list[int], dict]:
    """delta = delay(output) - delay(state); positive means the state detects first.

    A signal that never fires counts as W + 1, strictly later than any detection.
    Pairs where neither fires are dropped and counted.
    """
    deltas, counts = [], dict(both_censored=0, state_only_censored=0, output_only_censored=0, both_detected=0)
    for s, o in zip(delays_state, delays_output):
        if s is None and o is None:
            counts["both_censored"] += 1
            continue
        if s is None:
            counts["state_only_censored"] += 1
        elif o is None:
            counts["output_only_censored"] += 1
        else:
            counts["both_detected"] += 1
        deltas.append((W + 1 if o is None else o) - (W + 1 if s is None else s))
    return deltas, counts


def outcome(sign_p: float, median: float, ci_hi: float, n_pos: int, n_neg: int) -> str:
    """Pre-registered outcome mapping (PREREG_EARLY_DETECTION.md)."""
    if sign_p < 0.05:
        state_first = median > 0 or (median == 0 and n_pos > n_neg)
        if not state_first:
            return "output detects first: significant, the state gives no early warning"
        if median >= SESOI:
            return "early detection supported: state detects >= 2 rounds earlier"
        return "real early-detection effect below the size of interest (< 2 rounds)"
    if ci_hi < SESOI:
        return "effect of interest excluded: any state advantage is < 2 rounds"
    return "inconclusive"


def paired_test(deltas, n_boot: int = 10_000, seed: int = 0) -> dict:
    from scipy.stats import binomtest, wilcoxon

    x = np.asarray(deltas)
    nz = x[x != 0]
    n_pos, n_neg = int((nz > 0).sum()), int((nz < 0).sum())
    sign_p = float(binomtest(n_pos, len(nz)).pvalue) if len(nz) else 1.0
    boots = np.median(np.random.default_rng(seed).choice(x, size=(n_boot, len(x))), axis=1) if len(x) else np.zeros(1)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    med = float(np.median(x)) if len(x) else 0.0
    wil = float(wilcoxon(nz).pvalue) if len(nz) and np.any(nz != nz[0]) else None
    return dict(n=len(x), n_ties=int((x == 0).sum()), n_positive=n_pos, n_negative=n_neg, sign_test_p=sign_p,
                median_advantage=med, median_ci95=[lo, hi], wilcoxon_p=wil,
                outcome=outcome(sign_p, med, hi, n_pos, n_neg), deltas=x.tolist())


# --- Primary test (amended): normalized mean-curve timing ---------------------
#
# The operational detector above is infeasible as designed (power analysis in
# PREREG_EARLY_DETECTION.md). The primary test compares *when* each signal's
# across-run mean response reaches a fraction of its own rise, which removes
# the difference in response strength between the signals.

FRACTIONS = (0.10, 0.25, 0.50)
MARGIN = 2.0  # rounds; the effect of interest, in both directions


SMOOTH = 5  # centered moving average applied to each mean curve (same for both signals)
SUSTAIN = 3  # a crossing must hold for this many consecutive rounds


def smooth(curve: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    """Centered moving average; edges use the available rounds. Shifts both signals identically."""
    c = np.cumsum(np.insert(curve, 0, 0.0))
    i = np.arange(len(curve))
    lo, hi = np.maximum(0, i - k // 2), np.minimum(len(curve), i + k // 2 + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def crossfit_rise(x_a: np.ndarray, x_b: np.ndarray, base: float, W: int) -> float:
    """Rise without max-selection bias: the peak's location from one half of the runs, its value
    from the other half at that location, averaged over both directions (x_a, x_b: smoothed means)."""
    i_a, i_b = W + int(np.argmax(x_a[W:])), W + int(np.argmax(x_b[W:]))
    return float((x_b[i_a] + x_a[i_b]) / 2 - base)


def crossing_time(curve: np.ndarray, W: int, fraction: float, rise: float | None = None) -> float:
    """Rounds after the switch at which a mean curve first reaches `fraction` of its rise.

    `curve` covers [switch_at - W, switch_at + W] (length 2W + 1; index W is the switch).
    The curve is smoothed (SMOOTH). Baseline: mean over the W pre-switch rounds. Rise: `rise` if
    given (the cross-fitted rise, see `crossfit_rise`), else the post-switch max minus baseline.
    The crossing is the first post-switch round j at which the normalized curve is >= fraction for
    SUSTAIN consecutive rounds, linearly interpolated with round j - 1.
    """
    x = smooth(curve)
    base = x[:W].mean()
    n = (x - base) / (rise if rise is not None else x[W:].max() - base)
    above = n >= fraction
    for j in range(W, len(n) - SUSTAIN + 1):
        if above[j : j + SUSTAIN].all():
            if j == W or n[j - 1] >= fraction:
                return float(j - W)
            return float(j - 1 - W + (fraction - n[j - 1]) / (n[j] - n[j - 1]))
    return float(W)


def signal_crossings(runs: np.ndarray, W: int, fractions, weights: np.ndarray | None = None) -> list[float]:
    """Crossings of one signal's (weighted) mean curve, with a cross-fitted rise.
    The halves are the even- and odd-indexed runs."""
    w = np.full(len(runs), 1.0) if weights is None else weights
    halves = []
    for idx in (slice(0, None, 2), slice(1, None, 2)):
        ww = w[idx]
        halves.append(smooth((ww @ runs[idx]) / ww.sum()))
    full = (w @ runs) / w.sum()
    base = smooth(full)[:W].mean()
    rise = crossfit_rise(halves[0], halves[1], base, W)
    return [crossing_time(full, W, f, rise) for f in fractions]


def timing_differences(state: np.ndarray, output: np.ndarray, W: int, fractions=FRACTIONS,
                       weights: np.ndarray | None = None) -> np.ndarray:
    """delta_f = crossing(output) - crossing(state) for each fraction; positive = state first.
    `state`, `output`: (runs, 2W + 1) displacement windows, paired by run."""
    return np.array(signal_crossings(output, W, fractions, weights)) - np.array(signal_crossings(state, W, fractions, weights))


def bootstrap_timing(state: np.ndarray, output: np.ndarray, W: int, n_boot: int = 10_000, seed: int = 0,
                     fractions=FRACTIONS) -> dict:
    """Resample runs (paired across signals) and recompute mean curves, rises and crossings in each
    resample. Resampling is by integer weights, so the even/odd halves stay fixed."""
    rng = np.random.default_rng(seed)
    n = len(state)
    est = timing_differences(state, output, W, fractions)
    boots = np.empty((n_boot, len(fractions)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        if w[0::2].sum() == 0 or w[1::2].sum() == 0:
            w = np.ones(n)
        boots[b] = timing_differences(state, output, W, fractions, w)
    p = np.minimum(1.0, 2 * np.minimum((boots <= 0).mean(0), (boots >= 0).mean(0)))
    p = np.maximum(p, 1 / n_boot)
    return dict(estimate=est.tolist(), ci95=np.percentile(boots, [2.5, 97.5], axis=0).T.tolist(), p=p.tolist())


def holm(pvalues) -> list[float]:
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def timing_outcome(estimate: float, ci: list[float], p_adjusted: float, margin: float = MARGIN) -> str:
    """Outcome categories for one fraction, in both directions (PREREG_EARLY_DETECTION.md)."""
    lo, hi = ci
    if p_adjusted < 0.05:
        side = "state first" if estimate > 0 else "output first"
        size = f"by >= {margin:g} rounds" if abs(estimate) >= margin else f"by < {margin:g} rounds"
        return f"{side} {size}"
    if -margin < lo and hi < margin:
        return f"no difference of interest (95% CI within ±{margin:g} rounds)"
    return "inconclusive"
