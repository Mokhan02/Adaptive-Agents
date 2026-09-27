"""Iterated Rock-Paper-Scissors against a regime-switching opponent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ACTIONS = 3
ROCK, PAPER, SCISSORS = 0, 1, 2

# PAYOFF[a, b] = reward to the player choosing a against b.
PAYOFF = np.array(
    [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ],
    dtype=float,
)


def expected_payoffs(opp_dist: np.ndarray) -> np.ndarray:
    """Expected reward of each pure action against an opponent distribution."""
    return PAYOFF @ opp_dist


def best_response(opp_dist: np.ndarray) -> int:
    return int(np.argmax(expected_payoffs(opp_dist)))


def oracle_value(opp_dist: np.ndarray) -> float:
    """Expected reward of an agent that knows the opponent's true distribution."""
    return float(np.max(expected_payoffs(opp_dist)))


@dataclass
class RegimeSwitchOpponent:
    """Plays Strategy A, then transitions to Strategy B with no signal.

    The mixing weight on B is 0 before `switch_at`, rises linearly to 1 over
    `transition` rounds, and stays at 1 afterwards. `transition=0` is a hard
    discrete switch.
    """

    strategy_a: np.ndarray
    strategy_b: np.ndarray
    switch_at: int = 200
    transition: int = 0

    def __post_init__(self) -> None:
        self.strategy_a = _as_dist(self.strategy_a)
        self.strategy_b = _as_dist(self.strategy_b)
        if self.transition < 0:
            raise ValueError("transition must be >= 0")

    @classmethod
    def no_switch(cls, strategy: np.ndarray, switch_at: int = 200) -> "RegimeSwitchOpponent":
        """Control opponent that plays `strategy` throughout. `switch_at` is kept
        as an anchor so analysis windows line up with the real runs."""
        return cls(strategy, strategy, switch_at=switch_at, transition=0)

    @property
    def switch_end(self) -> int:
        """First round at which the opponent plays pure Strategy B."""
        return self.switch_at + self.transition

    def weight_b(self, t: int) -> float:
        if t < self.switch_at:
            return 0.0
        if t >= self.switch_end:
            return 1.0
        return (t - self.switch_at + 1) / (self.transition + 1)

    def distribution(self, t: int) -> np.ndarray:
        w = self.weight_b(t)
        return (1 - w) * self.strategy_a + w * self.strategy_b

    def act(self, t: int, rng: np.random.Generator) -> int:
        return int(rng.choice(N_ACTIONS, p=self.distribution(t)))


def _as_dist(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.shape != (N_ACTIONS,) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError(f"not a distribution over {N_ACTIONS} actions: {p}")
    return p


# Default regimes: rock-heavy then scissors-heavy, so the best response flips
# from paper to rock and a non-adapting agent is actively punished after the switch.
STRATEGY_A = np.array([0.6, 0.2, 0.2])
STRATEGY_B = np.array([0.2, 0.2, 0.6])
