"""Study 2 pipeline on planted signals (no model, no real seeds).

Noise traces mimic displacement series: a smoothed random process, the same
for switch and control runs. Planted responses are identical ramps in both
signals, with the state's onset shifted by a known number of rounds.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from regime.early_detection import (
    SESOI,
    binomial_interval,
    detection_delay,
    false_alarm_rate,
    outcome,
    paired_differences,
    paired_test,
    run_max,
    threshold,
)

A, W = 200, 86


def noise(rng, T=300, h=14):
    x = rng.normal(size=T + h).cumsum()
    return np.abs(x[h:] - x[:-h]) / np.sqrt(h)  # like a displacement over h rounds


def planted(rng, state_lead: int, amp: float = 3.0):
    t = np.arange(300)
    ramp = lambda onset: amp * np.clip((t - onset) / 10, 0, 1)
    onset = A + 10 + rng.integers(0, 10)
    return noise(rng) + ramp(onset - state_lead), noise(rng) + ramp(onset)


def delays(runs, tau_s, tau_o):
    return ([detection_delay(s, tau_s, A, W) for s, _ in runs], [detection_delay(o, tau_o, A, W) for _, o in runs])


@pytest.fixture(scope="module")
def taus():
    rng = np.random.default_rng(0)
    return threshold([run_max(noise(rng), A, W) for _ in range(1000)], 0.05)


def test_threshold_holds_false_alarm_rate_on_held_out_controls(taus):
    rng = np.random.default_rng(1)
    for alpha in (0.05, 0.01):
        tau = threshold([run_max(noise(rng), A, W) for _ in range(1000)], alpha)
        rate = false_alarm_rate([run_max(noise(rng), A, W) for _ in range(1000)], tau)
        lo, hi = binomial_interval(1000, alpha)
        assert lo <= rate <= hi, (alpha, rate)


@pytest.mark.parametrize("lead", [SESOI, -SESOI])
def test_recovers_planted_advantage_with_correct_sign(taus, lead):
    rng = np.random.default_rng(2)
    runs = [planted(rng, lead) for _ in range(300)]
    deltas, _ = paired_differences(*delays(runs, taus, taus), W)
    res = paired_test(deltas)
    assert res["sign_test_p"] < 0.05
    assert res["median_advantage"] == pytest.approx(lead, abs=1)
    assert (res["n_positive"] > res["n_negative"]) == (lead > 0)


def test_planted_null_rejects_at_most_nominally(taus):
    rng = np.random.default_rng(3)
    rejections = []
    for _ in range(150):
        runs = [planted(rng, 0) for _ in range(100)]
        deltas, _ = paired_differences(*delays(runs, taus, taus), W)
        rejections.append(paired_test(deltas, n_boot=10)["sign_test_p"] < 0.05)
    assert np.mean(rejections) <= 0.09


def test_censoring_rules():
    deltas, counts = paired_differences([3, None, None, 5], [5, 4, None, None], W)
    assert deltas == [2, 4 - (W + 1), (W + 1) - 5]
    assert counts == dict(both_censored=1, state_only_censored=1, output_only_censored=1, both_detected=1)


def test_detection_ignores_pre_switch_crossings():
    d = np.zeros(300)
    d[A - 5] = 10  # a false alarm before the window
    d[A + 7] = 10
    assert detection_delay(d, 1.0, A, W) == 7
    assert detection_delay(np.zeros(300), 1.0, A, W) is None


def test_outcome_mapping_covers_every_case():
    assert outcome(0.001, 3, 4, 80, 10).startswith("early detection supported")
    assert outcome(0.001, 1, 2, 80, 10).startswith("real early-detection effect below")
    assert outcome(0.001, 0, 1, 60, 20).startswith("real early-detection effect below")  # median 0, majority state-first
    assert outcome(0.001, -2, -1, 10, 80).startswith("output detects first")
    assert outcome(0.4, 0, 1, 40, 45).startswith("effect of interest excluded")
    assert outcome(0.4, 1, 3, 45, 40) == "inconclusive"
