import numpy as np
import pytest

pytest.importorskip("torch")

from regime.agents.online import ChangeAwareAgent, FineTuneAgent
from regime.env import STRATEGY_A, RegimeSwitchOpponent
from regime.metrics import excess_regret
from regime.pretrain import near_held_out
from regime.runner import run_episode
from regime.tuning import N_CONFIGS, sample_configs, score_episode, tuning_opponent

# Pretraining-style opponents for behavioral checks; tests never use A -> B.
P, Q = np.array([0.1, 0.8, 0.1]), np.array([0.1, 0.1, 0.8])


def opp():
    return RegimeSwitchOpponent(P, Q, switch_at=200)


@pytest.mark.parametrize("make", [lambda: FineTuneAgent(lr=3e-3, baseline_decay=0.99), lambda: ChangeAwareAgent()])
def test_agents_learn_and_adapt(make):
    logs = [run_episode(make(), opp(), 600, seed=s) for s in range(5)]
    pre = np.mean([l.instant_regret[150:200].mean() for l in logs])
    post = np.mean([l.instant_regret[500:].mean() for l in logs])
    uniform_regret = 0.7  # oracle value of both regimes
    assert pre < 0.5 * uniform_regret and post < 0.5 * uniform_regret


@pytest.mark.parametrize("make", [FineTuneAgent, ChangeAwareAgent])
def test_agents_are_reproducible_and_log_signals(make):
    a = run_episode(make(), opp(), 100, seed=3)
    b = run_episode(make(), opp(), 100, seed=3)
    np.testing.assert_array_equal(a.agent_actions, b.agent_actions)
    np.testing.assert_allclose(a.states, b.states)
    assert a.outputs.shape == (100, 3) and a.states.ndim == 2


def test_change_aware_temperature_rises_after_switch():
    agent = ChangeAwareAgent(forgetting=0.9, base_temperature=0.1, surprise_gain=2.0)
    log = run_episode(agent, opp(), 260, seed=0)
    temps = log.states[:, -1]
    assert temps[200:230].max() > temps[150:200].mean()


def test_change_aware_state_contains_outputs():
    log = run_episode(ChangeAwareAgent(), opp(), 50, seed=0)
    np.testing.assert_allclose(log.states[:, :3], log.outputs)


def test_tuning_opponents_follow_protocol():
    for i in range(50):
        o = tuning_opponent(i)
        assert o.switch_at == 200
        assert not any(near_held_out(o.distribution(t)) for t in range(0, 600, 5))
    np.testing.assert_allclose(tuning_opponent(7).strategy_a, tuning_opponent(7).strategy_a)


def test_config_sampling_is_fixed_and_in_range():
    for agent in ["in_context", "fine_tune", "change_aware"]:
        cs = sample_configs(agent)
        assert len(cs) == N_CONFIGS and cs == sample_configs(agent)
    assert all(1e-3 <= c["lr"] <= 1e-1 and c["window"] in (10, 25, 50) for c in sample_configs("fine_tune"))
    assert all(0.8 <= c["forgetting"] <= 0.99 for c in sample_configs("change_aware"))


def test_score_episode_matches_direct_computation():
    make = lambda: ChangeAwareAgent()
    log = run_episode(make(), tuning_opponent(0), 600, seed=5_000_000)
    assert score_episode(make, 0) == pytest.approx(excess_regret(log))


def test_fine_tune_step_matches_torch_adam():
    import torch
    from collections import deque

    agent = FineTuneAgent(lr=0.01, window=4)
    agent.reset(np.random.default_rng(0))
    moves = [0, 2, 1]
    agent.history = deque(moves, maxlen=4)
    agent._forward()
    W1, b1, W2, b2 = (torch.tensor(p, requires_grad=True) for p in agent.params)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=0.01)
    x = torch.tensor(agent._x)

    for a, r in [(1, 1.0), (0, -1.0), (2, 0.0), (1, 1.0)]:
        logits = W2 @ torch.tanh(W1 @ x + b1) + b2
        np.testing.assert_allclose(agent.output_scores(), logits.detach().numpy(), rtol=1e-10)
        loss = -(r - agent.baseline) * torch.log_softmax(logits, -1)[a]
        opt.zero_grad()
        loss.backward()
        opt.step()
        agent.observe(a, 0, r)
        agent.history = deque(moves, maxlen=4)  # keep the input fixed for the comparison
        agent._forward()
        for mine, theirs in zip(agent.params, [W1, b1, W2, b2]):
            np.testing.assert_allclose(mine, theirs.detach().numpy(), rtol=1e-8, atol=1e-12)
