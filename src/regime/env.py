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


class FictitiousPlayOpponent:
    """Best-responds to the agent's own recent play (study 3; ANALYSIS_PLAN.md).

    Before it has seen M agent moves it plays uniformly. After that, its mixed
    strategy is (1 - eps) * best response to the empirical distribution of the
    agent's last M sampled moves (uniform over tied best responses) + eps * uniform.
    No switch: `switch_at` and `switch_end` are 0 for EpisodeLog bookkeeping.
    """

    switch_at = 0
    switch_end = 0

    def __init__(self, window: int, eps: float = 0.1):
        if window < 1 or not 0 <= eps <= 1:
            raise ValueError("window >= 1 and 0 <= eps <= 1")
        self.window, self.eps = window, eps
        self.reset()

    def reset(self) -> None:
        from collections import deque

        self.history: deque[int] = deque(maxlen=self.window)

    def observe_agent(self, action: int) -> None:
        self.history.append(action)

    def distribution(self, t: int) -> np.ndarray:
        uniform = np.full(N_ACTIONS, 1 / N_ACTIONS)
        if len(self.history) < self.window:
            return uniform
        p = np.bincount(np.fromiter(self.history, int), minlength=N_ACTIONS) / self.window
        values = expected_payoffs(p)
        best = np.isclose(values, values.max())
        br = best / best.sum()
        return (1 - self.eps) * br + self.eps * uniform

    def act(self, t: int, rng: np.random.Generator) -> int:
        return int(rng.choice(N_ACTIONS, p=self.distribution(t)))
