import numpy as np
import pytest

from regime.env import STRATEGY_A, STRATEGY_B
from regime.pretrain import (
    EXCLUSION_EPS,
    MIN_SWITCH_TV,
    ScheduledOpponent,
    near_held_out,
    sample_pretraining_opponent,
    tv,
)


def test_test_strategies_and_their_rotations_are_held_out():
    assert near_held_out(STRATEGY_A) and near_held_out(STRATEGY_B)
    assert near_held_out(np.array([0.2, 0.6, 0.2]))
    assert near_held_out(np.array([0.65, 0.2, 0.15]))
    assert not near_held_out(np.array([0.8, 0.1, 0.1]))


def test_sampled_opponents_respect_the_spec():
    rng = np.random.default_rng(0)
    n_switches = []
    for _ in range(500):
        opp = sample_pretraining_opponent(rng)
        n_switches.append(len(opp.switch_at))
        # Every round's distribution, including mid-transition blends, stays outside the held-out balls.
        assert not any(near_held_out(opp.distribution(t)) for t in range(600))
        for p, q in zip(opp.strategies, opp.strategies[1:]):
            assert tv(p, q) >= MIN_SWITCH_TV
    assert set(n_switches) == {0, 1, 2, 3}


def test_scheduled_opponent_blends_and_holds():
    a, b, c = np.eye(3)
    opp = ScheduledOpponent([a, b, c], switch_at=[10, 30], transitions=[4, 0])
    np.testing.assert_allclose(opp.distribution(9), a)
    np.testing.assert_allclose(opp.distribution(10), 0.8 * a + 0.2 * b)
    np.testing.assert_allclose(opp.distribution(14), b)
    np.testing.assert_allclose(opp.distribution(29), b)
    np.testing.assert_allclose(opp.distribution(30), c)


def test_scheduled_opponent_rejects_overlapping_transitions():
    a, b, c = np.eye(3)
    with pytest.raises(ValueError):
        ScheduledOpponent([a, b, c], switch_at=[10, 12], transitions=[5, 0])


def test_exclusion_radius_is_below_sampling_resolution_of_context_window():
    # A rotation-of-A strategy estimated from 50 moves has SE ~0.07 on its
    # dominant action; eps sits at ~1.5 SE (see AGENT_SPECS.md).
    se = np.sqrt(0.6 * 0.4 / 50)
    assert 1.0 < EXCLUSION_EPS / se < 2.0
