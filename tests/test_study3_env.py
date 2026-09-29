"""Study 3's fictitious-play opponent. These tests double as its specification."""

import numpy as np
import pytest

from regime.agents import ConstantAgent, FrequencyAgent, UniformAgent
from regime.env import PAPER, ROCK, SCISSORS, FictitiousPlayOpponent
from regime.runner import run_episode


def test_always_rock_earns_exactly_minus_0_9_once_window_is_full():
    # eps-exploration is uniform over all three moves: 0.9 * (-1) + 0.1 * 0 = -0.9
    log = run_episode(ConstantAgent(ROCK), FictitiousPlayOpponent(window=20, eps=0.1), 200, seed=0)
    np.testing.assert_allclose(log.expected_rewards[:20], 0.0)  # uniform before M moves are seen
    np.testing.assert_allclose(log.expected_rewards[20:], -0.9)


def test_uniform_play_earns_exactly_zero():
    log = run_episode(UniformAgent(), FictitiousPlayOpponent(window=10, eps=0.1), 300, seed=1)
    np.testing.assert_allclose(log.expected_rewards, 0.0, atol=1e-12)


def test_ties_are_uniform_over_tied_best_responses():
    opp = FictitiousPlayOpponent(window=3, eps=0.0)
    for a in (ROCK, PAPER, SCISSORS):  # uniform history: every action ties at value 0
        opp.observe_agent(a)
    np.testing.assert_allclose(opp.distribution(3), [1 / 3] * 3)
    opp = FictitiousPlayOpponent(window=2, eps=0.0)
    opp.observe_agent(ROCK)
    opp.observe_agent(PAPER)  # p = (.5, .5, 0): paper (0.5) beats scissors (0) and rock (-0.5)
    np.testing.assert_allclose(opp.distribution(2), [0, 1, 0])


def test_best_response_uses_only_the_last_M_moves():
    opp = FictitiousPlayOpponent(window=3, eps=0.0)
    for a in [ROCK] * 10 + [SCISSORS] * 3:
        opp.observe_agent(a)
    np.testing.assert_allclose(opp.distribution(13), np.eye(3)[ROCK])  # rock beats scissors


def test_opponent_state_resets_between_episodes():
    opp = FictitiousPlayOpponent(window=5, eps=0.1)
    run_episode(ConstantAgent(ROCK), opp, 50, seed=0)
    log = run_episode(ConstantAgent(ROCK), opp, 50, seed=0)
    np.testing.assert_allclose(log.expected_rewards[:5], 0.0)


def test_scripted_opponents_are_unaffected_by_the_hook():
    from regime.env import STRATEGY_A, STRATEGY_B, RegimeSwitchOpponent

    log = run_episode(FrequencyAgent(window=20), RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B), 100, seed=3)
    assert log.expected_rewards.shape == (100,)


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        FictitiousPlayOpponent(window=0)


def test_study3_rule_helpers():
    from regime.study3 import cycle_period, separated, settle_round, trailing_entropy

    assert separated({"a": 0.0, "b": 0.1}, {"a": 0.01, "b": 0.01})
    assert not separated({"a": 0.0, "b": 0.02}, {"a": 0.01, "b": 0.01})  # 0.02 < 2 * 0.0141
    m = np.concatenate([np.linspace(-0.5, -0.1, 100), np.full(900, -0.1)])
    assert 90 <= settle_round(m) <= 100
    assert settle_round(np.zeros(1000)) == 0  # the absolute floor handles a zero band
    np.testing.assert_allclose(trailing_entropy(np.array([0, 1, 2] * 10), 3)[2:], np.log(3))
    assert cycle_period(np.array([0, 1, 2] * 100)) == 3
    assert cycle_period(np.zeros(300, dtype=int)) is None


def test_pair_outcome_names_the_agent_above():
    import sys

    sys.path.insert(0, "scripts")
    from study3 import pair_outcome

    assert pair_outcome("x", "y", -0.06, [-0.07, -0.05], 0.001) == "y above x by at least the effect of interest"
    assert pair_outcome("x", "y", 0.01, [0.005, 0.015], 0.01) == "x above y by less than the effect of interest"
    assert pair_outcome("x", "y", 0.001, [-0.01, 0.012], 0.5) == "equivalent within the margin"
    assert pair_outcome("x", "y", 0.01, [-0.01, 0.03], 0.5) == "inconclusive"


def test_bridge_opponent_is_scripted_then_fictitious_play():
    from regime.env import STRATEGY_A
    from regime.study3 import BridgeOpponent

    log = run_episode(ConstantAgent(ROCK), BridgeOpponent(window=5, eps=0.1, switch_at=50), 100, seed=0)
    np.testing.assert_allclose(log.expected_rewards[:50], STRATEGY_A @ np.array([0, -1, 1]))  # rock vs A
    np.testing.assert_allclose(log.expected_rewards[50:], -0.9)  # window already full at the switch


def test_study3b_outcomes_separate_small_changes_from_no_change():
    import sys

    sys.path.insert(0, "scripts")
    from study3 import change_outcome, prediction_status

    assert change_outcome(0.01, [0.004, 0.016], 0.01) == "increases by less than the margin"
    assert change_outcome(0.001, [-0.01, 0.012], 0.6) == "unchanged within the margin"
    assert change_outcome(-0.05, [-0.06, -0.04], 0.001) == "decreases by at least the margin"
    assert prediction_status("increases", "increases by less than the margin") == "confirmed"
    assert prediction_status("unchanged within the margin", "increases by less than the margin") == "not confirmed"
    assert prediction_status("decreases", "inconclusive") == "not confirmed (inconclusive)"
