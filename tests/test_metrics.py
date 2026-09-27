import numpy as np
import pytest

from regime.agents import FrequencyAgent, UniformAgent
from regime.env import STRATEGY_A, STRATEGY_B, RegimeSwitchOpponent
from regime.metrics import (
    cumulative_regret,
    excess_regret,
    moving_average,
    recovery_time,
    representational_drift,
)
from regime.runner import run_episode


def opponent(transition=0):
    return RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B, switch_at=200, transition=transition)


def test_moving_average_is_trailing():
    np.testing.assert_allclose(moving_average(np.array([2, 4, 6, 8]), 2), [2, 3, 5, 7])


def test_runs_are_reproducible():
    a = run_episode(FrequencyAgent(window=20), opponent(), 400, seed=3)
    b = run_episode(FrequencyAgent(window=20), opponent(), 400, seed=3)
    np.testing.assert_array_equal(a.agent_actions, b.agent_actions)
    np.testing.assert_array_equal(a.opp_actions, b.opp_actions)


def test_uniform_agent_regret_equals_oracle_value():
    log = run_episode(UniformAgent(), opponent(), 400, seed=0)
    np.testing.assert_allclose(log.instant_regret, 0.4)
    assert log.states is None
    assert excess_regret(log) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        recovery_time(log)
    gradual = run_episode(UniformAgent(), opponent(transition=100), 400, seed=0)
    assert excess_regret(gradual) == pytest.approx(0.0)


def test_excess_regret_ranks_agents_under_gradual_switch():
    def mean_excess(make):
        return np.mean([excess_regret(run_episode(make(), opponent(transition=100), 600, s)) for s in range(10)])

    assert mean_excess(lambda: FrequencyAgent(window=20)) < mean_excess(lambda: FrequencyAgent())


def test_windowed_agent_recovers_and_full_history_agent_is_slower():
    windowed = [recovery_time(run_episode(FrequencyAgent(window=20), opponent(), 600, s)) for s in range(10)]
    full = [recovery_time(run_episode(FrequencyAgent(), opponent(), 600, s)) for s in range(10)]
    assert all(r is not None for r in windowed)
    assert np.median(windowed) < 50
    # The full-history agent must wait for B counts to outweigh ~200 rounds of A.
    assert np.median([r if r is not None else 10**6 for r in full]) > 2 * np.median(windowed)


def test_regret_accumulates_mostly_after_switch_for_non_forgetting_agent():
    log = run_episode(FrequencyAgent(), opponent(), 400, seed=0)
    cr = cumulative_regret(log)
    assert cr[-1] - cr[199] > cr[199]


def test_drift():
    s = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(representational_drift(s)[1:], [0.0, 1.0])
    np.testing.assert_allclose(representational_drift(s, "l2")[1:], [0.0, np.sqrt(2)])
    assert np.isnan(representational_drift(s)[0])


def test_gradual_switch_recovery_measured_from_switch_end():
    log = run_episode(FrequencyAgent(window=20), opponent(transition=100), 600, seed=0)
    assert log.switch_end == 300
    assert recovery_time(log) is not None


def test_outputs_are_logged_after_observe_and_move_before_argmax_behavior():
    log = run_episode(FrequencyAgent(), opponent(), 400, seed=0)
    assert log.outputs.shape == (400, 3)
    # Output scores start shifting right after the switch, well before the
    # greedy action flips from paper to rock.
    first_flip = 200 + int(np.argmax(log.agent_actions[200:] == 0))
    moved = np.linalg.norm(log.outputs[first_flip - 1] - log.outputs[199])
    assert moved > 0.1
