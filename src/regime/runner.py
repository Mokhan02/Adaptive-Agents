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
) -> EpisodeLog:
    rng = np.random.default_rng(seed)
    agent_rng, opp_rng = rng.spawn(2)
    agent.reset(agent_rng)

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
        b = opponent.act(t, opp_rng)
        r = PAYOFF[a, b]

        agent.observe(a, b, r)

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
