import numpy as np
import pytest

from regime.env import (
    PAPER,
    ROCK,
    STRATEGY_A,
    STRATEGY_B,
    RegimeSwitchOpponent,
    best_response,
    oracle_value,
)


def test_best_responses_to_default_regimes():
    assert best_response(STRATEGY_A) == PAPER
    assert best_response(STRATEGY_B) == ROCK
    assert oracle_value(STRATEGY_A) == pytest.approx(0.4)
    assert oracle_value(STRATEGY_B) == pytest.approx(0.4)


def test_hard_switch():
    opp = RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B, switch_at=10, transition=0)
    np.testing.assert_allclose(opp.distribution(9), STRATEGY_A)
    np.testing.assert_allclose(opp.distribution(10), STRATEGY_B)
    assert opp.switch_end == 10


def test_gradual_switch_is_linear_and_ends_on_b():
    opp = RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B, switch_at=10, transition=4)
    weights = [opp.weight_b(t) for t in range(9, 16)]
    np.testing.assert_allclose(weights, [0, 0.2, 0.4, 0.6, 0.8, 1, 1])
    np.testing.assert_allclose(opp.distribution(opp.switch_end), STRATEGY_B)


def test_empirical_frequencies_match_regimes():
    opp = RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B, switch_at=5000)
    rng = np.random.default_rng(0)
    moves = np.array([opp.act(t, rng) for t in range(10000)])
    np.testing.assert_allclose(np.bincount(moves[:5000], minlength=3) / 5000, STRATEGY_A, atol=0.02)
    np.testing.assert_allclose(np.bincount(moves[5000:], minlength=3) / 5000, STRATEGY_B, atol=0.02)


def test_rejects_invalid_distribution():
    with pytest.raises(ValueError):
        RegimeSwitchOpponent([0.5, 0.5, 0.5], STRATEGY_B)


def test_no_switch_control_keeps_anchor_and_never_changes():
    opp = RegimeSwitchOpponent.no_switch(STRATEGY_A, switch_at=200)
    assert opp.switch_at == 200
    for t in (0, 199, 200, 599):
        np.testing.assert_allclose(opp.distribution(t), STRATEGY_A)
