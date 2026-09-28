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


# --- amended primary test: normalized mean-curve timing ------------------------

from regime.early_detection import FRACTIONS, bootstrap_timing, crossing_time, holm, timing_differences, timing_outcome


def rise(W, onset, amp, width=12):
    t = np.arange(-W, W + 1)
    return amp * np.clip((t - onset) / width, 0, 1)


def test_crossing_time_interpolates_and_is_amplitude_invariant():
    c = rise(W, onset=5, amp=1.0)
    assert crossing_time(c, W, 0.5) == pytest.approx(5 + 6)
    assert crossing_time(3.7 * c + 2.0, W, 0.5) == pytest.approx(11)  # scale and offset don't matter
    assert crossing_time(c, W, 0.25) == pytest.approx(5 + 3)


def planted_curves(rng, n, lead, amp_state, amp_out=1.0):
    s = np.stack([rise(W, 8 - lead, amp_state) + 0.3 * rng.normal(size=2 * W + 1) for _ in range(n)])
    o = np.stack([rise(W, 8, amp_out) + 0.3 * rng.normal(size=2 * W + 1) for _ in range(n)])
    return s, o


@pytest.mark.parametrize("amp_state", [0.4, 1.0, 2.5])
def test_timing_null_is_invariant_to_amplitude(amp_state):
    """Smoothing, a sustained crossing and a cross-fitted rise keep the null unbiased whatever the
    state's amplitude relative to the output's (the naive own-peak rule was biased both ways)."""
    rng = np.random.default_rng(10)
    fp, est = [], []
    for _ in range(60):
        s, o = planted_curves(rng, 200, 0, amp_state)
        r = bootstrap_timing(s, o, W, n_boot=200, seed=int(rng.integers(1e9)))
        fp.append(np.array(r["p"]) < 0.05)
        est.append(r["estimate"])
    fp, est = np.mean(fp, 0), np.mean(est, 0)
    assert np.all(fp <= [0.15, 0.12, 0.12]), fp  # nominal 5%; 60 experiments, 200 resamples
    assert np.all(np.abs(est) < 0.3), est


def test_crossfit_rise_removes_max_selection_bias():
    from regime.early_detection import crossfit_rise, smooth

    rng = np.random.default_rng(12)
    true = rise(W, 8, 0.4)
    naive, fitted = [], []
    for _ in range(300):
        runs = np.stack([true + 0.3 * rng.normal(size=2 * W + 1) for _ in range(100)])
        full = smooth(runs.mean(0))
        base = full[:W].mean()
        naive.append(full[W:].max() - base)
        fitted.append(crossfit_rise(smooth(runs[0::2].mean(0)), smooth(runs[1::2].mean(0)), base, W))
    assert np.mean(naive) > 0.4 + 0.01  # max of a noisy curve is inflated
    assert np.mean(fitted) == pytest.approx(0.4, abs=0.01)


def test_timing_recovers_planted_lead_with_sign():
    rng = np.random.default_rng(11)
    for lead in (2, -2):
        s, o = planted_curves(rng, 400, lead, 1.0)
        est = timing_differences(s, o, W)
        assert np.all(np.abs(est - lead) < 1.0), est


def test_holm_and_timing_outcomes():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert timing_outcome(3.0, [2.2, 3.8], 0.001) == "state first by >= 2 rounds"
    assert timing_outcome(-1.0, [-1.6, -0.4], 0.01) == "output first by < 2 rounds"
    assert timing_outcome(-3.0, [-4, -2.5], 0.001) == "output first by >= 2 rounds"
    assert timing_outcome(0.3, [-1.5, 1.8], 0.6).startswith("no difference of interest")
    assert timing_outcome(0.3, [-2.5, 3.0], 0.6) == "inconclusive"
