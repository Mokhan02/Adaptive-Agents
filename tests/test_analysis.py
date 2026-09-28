"""Analysis pipeline on planted signals (never on real seeds).

State and behavior series get a ramp at a known onset; the behavior's
onset is shifted by a planted lag. The pipeline must recover the lag with
the pre-registered sign (positive = state moves first) and hold its false-
positive rate when there is no lag.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from regime.analysis import (
    Calibration,
    censored_quantile,
    lag_estimate,
    settle_round,
    signals,
    run_test_a,
    peak_state_displacement,
    run_test_b,
)
from regime.runner import EpisodeLog

T, SWITCH, H, W, L = 400, 200, 10, 60, 30
CAL = Calibration("planted", t_w=20, median_recovery=20, p90_recovery=30, n_censored=0, runnable=True,
                  h=H, W=W, L=L, switch_at=SWITCH, n_rounds=T, state_mean=[0.0] * 8, state_std=[1.0] * 8,
                  control_mean_regret=[])


def ramp(t, onset, width=15):
    return np.clip((t - onset) / width, 0, 1)[:, None]


def planted_log(rng, lead: int, switch: bool = True) -> EpisodeLog:
    """State ramps at `onset`; behavior ramps at `onset + lead` (lead > 0: state first)."""
    t = np.arange(T)
    onset = SWITCH + rng.integers(0, 10)
    v, u = rng.normal(size=8), rng.normal(size=3)
    s_ramp = ramp(t, onset) if switch else 0 * ramp(t, onset)
    o_ramp = ramp(t, onset + lead) if switch else 0 * ramp(t, onset)
    states = 3 * s_ramp * v + 0.3 * rng.normal(size=(T, 8))
    outputs = 3 * o_ramp * u + 0.3 * rng.normal(size=(T, 3))
    e = np.empty(0)
    return EpisodeLog(e, e, e, e, e, e, outputs, states, SWITCH, SWITCH)


def lags_for(lead: int, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d_s, d_o = signals(planted_log(rng, lead), CAL)
        out.append(lag_estimate(d_s, d_o, SWITCH, W, L))
    return out


@pytest.mark.parametrize("lead", [8, -8])
def test_recovers_planted_lag_with_correct_sign(lead):
    res = run_test_b(lags_for(lead, 100, seed=1))
    assert abs(res["median_lag"] - lead) <= 1
    assert res["significant"]
    if lead > 0:
        assert res["n_positive"] > 90
    else:
        assert res["n_negative"] > 90


def test_null_false_positive_rate_is_at_most_nominal():
    rng = np.random.default_rng(2)
    rejections = [run_test_b(lags_for(0, 100, seed=int(rng.integers(2**31))), n_boot=10)["significant"] for _ in range(200)]
    rate = np.mean(rejections)
    assert rate <= 0.09  # nominal 5%; the exact sign test is slightly conservative


def test_lag_tie_between_plus_and_minus_k_is_zero():
    # A symmetric bump in both series at the same place: +k and -k correlate equally.
    s = np.zeros(2 * W + 1)
    s[W] = 1.0
    d = np.full(SWITCH + W + 1, np.nan)
    d[SWITCH - W :] = s
    assert lag_estimate(d, d.copy(), SWITCH, W, L) == 0


def test_a_separates_planted_switch_from_control():
    rng = np.random.default_rng(3)
    sw = [peak_state_displacement(signals(planted_log(rng, 0), CAL)[0], SWITCH, W) for _ in range(100)]
    ct = [peak_state_displacement(signals(planted_log(rng, 0, switch=False), CAL)[0], SWITCH, W) for _ in range(100)]
    assert run_test_a(sw, ct)["passed"]
    ct2 = [peak_state_displacement(signals(planted_log(rng, 0, switch=False), CAL)[0], SWITCH, W) for _ in range(100)]
    assert not run_test_a(ct2, ct)["passed"]


def test_censored_quantiles():
    times = [10, 20, 30, None, None]
    assert censored_quantile(times, 0.5) == 30
    assert censored_quantile(times, 0.9) is None  # 40% censored
    assert censored_quantile([5] * 9 + [None], 0.9) == 5  # exactly 10% censored is still observed


def test_settle_round_uses_relative_drop():
    m = np.concatenate([np.linspace(0.4, 0.02, 50), np.full(200, 0.02) + 0.005 * np.sin(np.arange(200))])
    # Band is 10% of the 0.38 drop (0.038), so small wiggles around 0.02 don't delay settling.
    assert 40 <= settle_round(m) <= 50


def test_calibration_standardization_uses_rounds_after_late_warm_up():
    from regime.analysis import calibrate

    rng = np.random.default_rng(4)
    e = np.empty(0)

    def log(switch):
        regret = np.concatenate([np.linspace(0.4, 0.0, 300), np.zeros(300)])  # settles at ~round 300
        states = rng.normal(loc=2.0, size=(600, 4))
        expected = 0.4 - regret
        return EpisodeLog(e, e, e, expected, np.full(600, 0.4), e, np.zeros((600, 3)), states, 200, 200)

    cal = calibrate("late", [log(True) for _ in range(5)], [log(False) for _ in range(5)])
    assert cal.t_w > 200
    assert np.all(np.isfinite(cal.state_mean)) and np.all(np.isfinite(cal.state_std))
    np.testing.assert_allclose(cal.state_mean, 2.0, atol=0.1)
