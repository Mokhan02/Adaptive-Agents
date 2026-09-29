"""Partial observability (lag, noise) in the runner. With both off, earlier results reproduce exactly."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from regime.agents import ConstantAgent, FrequencyAgent
from regime.agents.base import Agent
from regime.env import PAYOFF, ROCK, FictitiousPlayOpponent, expected_payoffs
from regime.runner import run_episode

RESULTS = Path("results/study3/results.json")


@pytest.mark.skipif(not RESULTS.exists(), reason="study 3 results not present")
@pytest.mark.parametrize("name", ["in_context", "change_aware", "fine_tune"])
def test_no_lag_no_noise_reproduces_study3_trajectories(name):
    pytest.importorskip("torch")
    from regime.analysis import make_agent

    r = json.loads(RESULTS.read_text())
    T_w = r["T_w"]
    for i, ex in enumerate(r["example_trajectories"][name]):
        log = run_episode(make_agent(name), FictitiousPlayOpponent(r["opponent"]["M"], r["opponent"]["eps"]),
                          r["n_rounds"], 60000 + i)
        assert log.agent_actions[T_w : T_w + 60].tolist() == ex["actions"]
        assert log.opp_actions[T_w : T_w + 60].tolist() == ex["opponent"]


class Recorder(Agent):
    """Uniform player that records what it is shown."""

    def reset(self, rng):
        super().reset(rng)
        self.seen = []

    def policy(self):
        return np.full(3, 1 / 3)

    def observe(self, my_action, opp_action, reward):
        self.seen.append((my_action, opp_action, reward))

    def output_scores(self):
        return np.zeros(3)


def test_agent_lag_delivers_feedback_late_and_in_order():
    agent = Recorder()
    log = run_episode(agent, FictitiousPlayOpponent(5, 0.1), 50, seed=0, agent_lag=2)
    assert len(agent.seen) == 48  # the last 2 rounds' feedback never arrives within the episode
    for t, (a, b, r) in enumerate(agent.seen):
        assert (a, b, r) == (log.agent_actions[t], log.opp_actions[t], log.rewards[t])


def test_opponent_lag_uses_stale_agent_moves():
    # Always-rock: the opponent's window fills 3 rounds later, so rounds 5-7 stay uniform (reward 0).
    log = run_episode(ConstantAgent(ROCK), FictitiousPlayOpponent(5, 0.1), 20, seed=0, opponent_lag=3)
    np.testing.assert_allclose(log.expected_rewards[:8], 0.0)
    np.testing.assert_allclose(log.expected_rewards[8:], -0.9)


def test_agent_noise_corrupts_observed_moves_at_two_thirds_the_rate():
    agent = Recorder()
    q = 0.3
    log = run_episode(agent, FictitiousPlayOpponent(5, 0.1), 20_000, seed=1, agent_noise=q)
    seen = np.array([b for _, b, _ in agent.seen])
    assert np.mean(seen != log.opp_actions) == pytest.approx(q * 2 / 3, abs=0.01)  # uniform draw can match
    assert all(r == PAYOFF[a, b] for (a, _, r), b in zip(agent.seen, log.opp_actions))  # rewards are true


def exact_always_rock_under_opponent_noise(q: float, M: int = 5, eps: float = 0.1) -> float:
    """Expected reward of always-rock when the opponent sees each move as rock w.p. 1 - 2q/3 and as
    paper or scissors w.p. q/3 each, independently: enumerate all 3^M windows."""
    p_obs = np.array([1 - 2 * q / 3, q / 3, q / 3])
    total = 0.0
    for window in itertools.product(range(3), repeat=M):
        prob = np.prod(p_obs[list(window)])
        values = expected_payoffs(np.bincount(window, minlength=3) / M)
        best = np.isclose(values, values.max())
        opp = (1 - eps) * best / best.sum() + eps / 3
        total += prob * float(PAYOFF[ROCK] @ opp)
    return total


@pytest.mark.parametrize("q", [0.1, 0.3])
def test_opponent_noise_matches_exact_enumeration(q):
    exact = exact_always_rock_under_opponent_noise(q)
    log = run_episode(ConstantAgent(ROCK), FictitiousPlayOpponent(5, 0.1), 40_000, seed=2, opponent_noise=q)
    assert log.expected_rewards[5:].mean() == pytest.approx(exact, abs=0.01)
    assert exact > -0.9  # noise can only make the opponent's read of always-rock worse


def test_fine_tune_delayed_update_uses_decision_time_input():
    from regime.agents.online import FineTuneAgent

    agent = FineTuneAgent(lr=0.01, window=4)
    agent.reset(np.random.default_rng(0))
    x0 = agent._x.copy()
    agent.on_action(1)
    agent.history.extend([0, 2])  # the input changes before the feedback arrives
    agent._forward()
    expected = agent.gradients(1, 1.0 - agent.baseline, x0)
    before = [p.copy() for p in agent.params]
    agent.observe(1, 0, 1.0)
    # Adam's first step moves each parameter by -lr * sign(grad) where grad != 0
    for p0, p1, g in zip(before, agent.params, expected):
        np.testing.assert_allclose(p1 - p0, -0.01 * np.sign(g), atol=1e-6)
