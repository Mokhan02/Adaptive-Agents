from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regime.agents.base import Agent
from regime.env import PAYOFF, RegimeSwitchOpponent, expected_payoffs, oracle_value


@dataclass
class EpisodeLog:
    """Per-round record of one episode. Arrays are indexed by round."""

    agent_actions: np.ndarray  # (T,)
    opp_actions: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,) realized reward
    expected_rewards: np.ndarray  # (T,) agent's expected reward given its policy and the true opponent dist
    oracle_rewards: np.ndarray  # (T,) expected reward of the best response to the true opponent dist
    policies: np.ndarray  # (T, 3)
    outputs: np.ndarray  # (T, 3) output scores *after* observing round t (the behavior signal for lead/lag)
    states: np.ndarray | None  # (T, d) internal state *after* observing round t, if the agent tracks one
    switch_at: int
    switch_end: int

    @property
    def instant_regret(self) -> np.ndarray:
        """Expected-reward gap to the oracle. Uses expectations rather than
        realized rewards so regret is not dominated by sampling noise."""
        return self.oracle_rewards - self.expected_rewards


def run_episode(
    agent: Agent,
    opponent: RegimeSwitchOpponent,
    n_rounds: int,
    seed: int,
    agent_lag: int = 0,
    opponent_lag: int = 0,
    agent_noise: float = 0.0,
    opponent_noise: float = 0.0,
) -> EpisodeLog:
    """Play one episode.

    Partial observability (study 3; all default to off, which reproduces earlier results exactly):
    - agent_lag: the agent receives round t's (own action, opponent move, reward) after round t + lag.
    - opponent_lag: a reactive opponent sees the agent's round-t move after round t + lag.
    - agent_noise / opponent_noise: with this probability the *observed* move is replaced by a uniform
      draw over all three moves (it can come out unchanged). Rewards are never corrupted.
    """
    from collections import deque

    rng = np.random.default_rng(seed)
    agent_rng, opp_rng = rng.spawn(2)
    # Separate streams, spawned only when used, so the defaults leave every other stream untouched.
    noise_a, noise_o = rng.spawn(2) if (agent_noise or opponent_noise) else (None, None)
    to_agent: deque = deque()
    to_opponent: deque = deque()
    agent.reset(agent_rng)
    if hasattr(opponent, "reset"):  # reactive opponents carry per-episode state
        opponent.reset()

    agent_actions = np.empty(n_rounds, dtype=int)
    opp_actions = np.empty(n_rounds, dtype=int)
    rewards = np.empty(n_rounds)
    expected = np.empty(n_rounds)
    oracle = np.empty(n_rounds)
    policies = np.empty((n_rounds, PAYOFF.shape[0]))
    outputs = np.empty((n_rounds, PAYOFF.shape[0]))
    states: list[np.ndarray] = []

    for t in range(n_rounds):
        pi = agent.policy()
        opp_dist = opponent.distribution(t)
        a = int(agent_rng.choice(len(pi), p=pi))
        agent.on_action(a)
        b = opponent.act(t, opp_rng)
        r = PAYOFF[a, b]

        b_seen = b if not agent_noise or noise_a.random() >= agent_noise else int(noise_a.integers(3))
        to_agent.append((a, b_seen, r))
        while len(to_agent) > agent_lag:
            agent.observe(*to_agent.popleft())
        if hasattr(opponent, "observe_agent"):  # a reactive opponent sees the agent's move
            a_seen = a if not opponent_noise or noise_o.random() >= opponent_noise else int(noise_o.integers(3))
            to_opponent.append(a_seen)
            while len(to_opponent) > opponent_lag:
                opponent.observe_agent(to_opponent.popleft())

        agent_actions[t], opp_actions[t], rewards[t] = a, b, r
        expected[t] = float(pi @ expected_payoffs(opp_dist))
        oracle[t] = oracle_value(opp_dist)
        policies[t] = pi
        outputs[t] = agent.output_scores()
        s = agent.internal_state()
        if s is not None:
            states.append(np.asarray(s, dtype=float).ravel().copy())

    return EpisodeLog(
        agent_actions=agent_actions,
        opp_actions=opp_actions,
        rewards=rewards,
        expected_rewards=expected,
        oracle_rewards=oracle,
        policies=policies,
        outputs=outputs,
        states=np.stack(states) if states else None,
        switch_at=opponent.switch_at,
        switch_end=opponent.switch_end,
    )
