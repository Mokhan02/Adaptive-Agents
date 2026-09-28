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
