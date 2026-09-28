"""Simple reference agents for sanity-checking the pipeline.

These are not the three agents under comparison; they bracket the expected
behavior (never adapts / adapts via a sliding window) so the metrics can be
checked against known answers.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from regime.agents.base import Agent
from regime.env import N_ACTIONS, expected_payoffs


class UniformAgent(Agent):
    """Plays uniformly at random. Zero expected reward, never adapts."""

    name = "uniform"

    def policy(self) -> np.ndarray:
        return np.full(N_ACTIONS, 1 / N_ACTIONS)

    def observe(self, my_action: int, opp_action: int, reward: float) -> None:
        pass

    def output_scores(self) -> np.ndarray:
        return np.zeros(N_ACTIONS)


class ConstantAgent(Agent):
    """Always plays the same move: maximally exploitable (a study 3 reference)."""

    def __init__(self, action: int = 0):
        self.action = action
        self.name = f"always_{action}"

    def policy(self) -> np.ndarray:
        return np.eye(N_ACTIONS)[self.action]

    def observe(self, my_action: int, opp_action: int, reward: float) -> None:
        pass

    def output_scores(self) -> np.ndarray:
        return np.eye(N_ACTIONS)[self.action]


class FrequencyAgent(Agent):
    """Best-responds to the empirical opponent distribution.

    With `window=None` it counts all history (slow to adapt after a switch);
    with a finite window it only uses the last `window` opponent moves.
    `temperature` softens the best response; 0 means greedy.
    """

    def __init__(self, window: int | None = None, temperature: float = 0.0, prior: float = 1.0):
        self.window = window
        self.temperature = temperature
        self.prior = prior
        self.name = "frequency_all" if window is None else f"frequency_w{window}"

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        self.history: deque[int] = deque(maxlen=self.window)

    def belief(self) -> np.ndarray:
        counts = np.full(N_ACTIONS, self.prior)
        for a in self.history:
            counts[a] += 1
        return counts / counts.sum()

    def output_scores(self) -> np.ndarray:
        """Expected payoff of each action under the current belief."""
        return expected_payoffs(self.belief())

    def policy(self) -> np.ndarray:
        values = self.output_scores()
        if self.temperature == 0:
            best = np.flatnonzero(values == values.max())
            p = np.zeros(N_ACTIONS)
            p[best] = 1 / len(best)
            return p
        z = np.exp((values - values.max()) / self.temperature)
        return z / z.sum()

    def observe(self, my_action: int, opp_action: int, reward: float) -> None:
        self.history.append(opp_action)

    def internal_state(self) -> np.ndarray:
        return self.belief()
