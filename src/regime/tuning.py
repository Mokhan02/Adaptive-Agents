"""Hyperparameter tuning protocol (AGENT_SPECS.md, amendment "tuning protocol").

Every configuration of every agent plays the same 200 single-switch episodes
drawn from the pretraining distribution and is scored by mean
oracle-normalized excess regret.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from regime.agents.base import Agent
from regime.env import RegimeSwitchOpponent
from regime.metrics import excess_regret
from regime.pretrain import MAX_TRANSITION, MIN_SWITCH_TV, _sample_strategy, segment_near_held_out, tv
from regime.runner import run_episode

TUNING_SEED_BASE = 5_000_000
N_TUNING_EPISODES = 200
N_CONFIGS = 30
CONFIG_SEED_BASE = 6_000_000
AGENT_INDEX = {"in_context": 0, "fine_tune": 1, "change_aware": 2}
SWITCH_AT, N_ROUNDS = 200, 600


def tuning_opponent(i: int) -> RegimeSwitchOpponent:
    rng = np.random.default_rng(TUNING_SEED_BASE + i)
    transition = 0 if rng.random() < 0.5 else int(rng.integers(1, MAX_TRANSITION + 1))
    p = _sample_strategy(rng)
    while True:
        q = _sample_strategy(rng)
        if tv(p, q) >= MIN_SWITCH_TV and not (transition > 0 and segment_near_held_out(p, q)):
            return RegimeSwitchOpponent(p, q, switch_at=SWITCH_AT, transition=transition)


def score_episode(make_agent: Callable[[], Agent], i: int) -> float:
    log = run_episode(make_agent(), tuning_opponent(i), N_ROUNDS, seed=TUNING_SEED_BASE + i)
    return excess_regret(log)


def score_in_context_fast(model, temperature: float, i: int, device: str = "cpu") -> float:
    """Same score as `score_episode(lambda: InContextAgent(model, temperature), i)`,
    in one batched forward pass.

    Valid because the opponent ignores the agent: the opponent's moves (drawn
    from its own spawned RNG, as in `run_episode`) fix the agent's
    predictions, and regret uses the policy's expected reward, not its
    sampled actions.
    """
    import torch

    from regime.env import PAYOFF, expected_payoffs, oracle_value
    from regime.runner import EpisodeLog

    opp = tuning_opponent(i)
    _, opp_rng = np.random.default_rng(TUNING_SEED_BASE + i).spawn(2)
    moves = np.array([opp.act(t, opp_rng) for t in range(N_ROUNDS)])
    K = model.cfg.context

    with torch.no_grad():
        # Causal model: logits at position j of the first window predict round j + 1.
        first, _ = model(torch.as_tensor(moves[None, :K], device=device))
        windows = np.lib.stride_tricks.sliding_window_view(moves[: N_ROUNDS - 1], K)[1:]  # windows ending at rounds K..N-2
        rest, _ = model(torch.as_tensor(windows.copy(), device=device))
        logits = torch.cat([first[0], rest[:, -1]]).double().cpu()
    preds = np.concatenate([np.full((1, 3), 1 / 3), torch.softmax(logits, -1).numpy()])[:N_ROUNDS]

    z = preds @ PAYOFF.T / temperature
    pi = np.exp(z - z.max(1, keepdims=True))
    pi /= pi.sum(1, keepdims=True)
    dists = np.stack([opp.distribution(t) for t in range(N_ROUNDS)])
    expected = np.einsum("ta,ta->t", pi, dists @ PAYOFF.T)
    oracle = np.array([oracle_value(d) for d in dists])
    empty = np.empty(0)
    log = EpisodeLog(empty, moves, empty, expected, oracle, pi, preds @ PAYOFF.T, None, opp.switch_at, opp.switch_end)
    return excess_regret(log)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_configs(agent: str) -> list[dict]:
    rng = np.random.default_rng(CONFIG_SEED_BASE + AGENT_INDEX[agent])
    configs = []
    for _ in range(N_CONFIGS):
        if agent == "in_context":
            c = dict(lr=_log_uniform(rng, 1e-4, 1e-3), steps=int(rng.choice([2500, 5000, 10000, 20000])),
                     temperature=_log_uniform(rng, 0.02, 0.5))
        elif agent == "fine_tune":
            c = dict(lr=_log_uniform(rng, 1e-3, 1e-1), baseline_decay=float(rng.uniform(0.8, 0.99)),
                     window=int(rng.choice([10, 25, 50])))
        elif agent == "change_aware":
            c = dict(forgetting=float(rng.uniform(0.8, 0.99)), base_temperature=_log_uniform(rng, 0.02, 0.5),
                     surprise_gain=float(rng.uniform(0, 5)))
        else:
            raise ValueError(agent)
        configs.append(c)
    return configs
