"""Pretraining opponents for the in-context agent (see AGENT_SPECS.md).

Pretraining episodes randomize everything the test fixes (number, timing and
abruptness of switches) and exclude every strategy near the test strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regime.env import N_ACTIONS, STRATEGY_A, STRATEGY_B, _as_dist

# RPS is invariant under cyclic relabeling of actions, so all rotations of the
# test strategy are the same problem. B is one of them.
HELD_OUT = np.stack([np.roll(STRATEGY_A, k) for k in range(N_ACTIONS)])
assert any(np.allclose(STRATEGY_B, h) for h in HELD_OUT)

EXCLUSION_EPS = 0.10  # total-variation radius around each held-out strategy
MIN_SWITCH_TV = 0.20  # consecutive regimes must differ by at least this much
MAX_SWITCHES = 3
MAX_TRANSITION = 200
EDGE_MARGIN = 50  # no switch starts in the first or last 50 rounds
MIN_GAP = 30  # rounds of pure regime between the end of one transition and the next switch


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def near_held_out(p: np.ndarray, eps: float = EXCLUSION_EPS) -> bool:
    return min(tv(p, h) for h in HELD_OUT) < eps


def segment_tv(p: np.ndarray, q: np.ndarray, h: np.ndarray) -> float:
    """Exact minimum TV distance from h to any blend (1 - w) p + w q, w in [0, 1].

    TV along the segment is convex and piecewise linear in w, with kinks only
    where a coordinate of the blend crosses h, so the minimum is at an endpoint
    or one of those kinks.
    """
    d = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        kinks = np.where(d != 0, (h - p) / d, np.nan)
    ws = np.concatenate([[0.0, 1.0], kinks[(kinks > 0) & (kinks < 1)]])
    return min(tv(p + w * d, h) for w in ws)


def segment_near_held_out(p: np.ndarray, q: np.ndarray, eps: float = EXCLUSION_EPS) -> bool:
    """Whether a gradual blend from p to q passes within eps of a held-out strategy."""
    return min(segment_tv(p, q, h) for h in HELD_OUT) < eps


@dataclass
class ScheduledOpponent:
    """Opponent with any number of regime switches, each hard or gradual.

    Switch j starts at `switch_at[j]` and blends from `strategies[j]` to
    `strategies[j + 1]` over `transitions[j]` rounds (0 = hard switch).
    """

    strategies: list[np.ndarray]
    switch_at: list[int]
    transitions: list[int]

    def __post_init__(self) -> None:
        self.strategies = [_as_dist(s) for s in self.strategies]
        if not (len(self.strategies) == len(self.switch_at) + 1 == len(self.transitions) + 1):
            raise ValueError("need one more strategy than switches, and one transition per switch")
        for j in range(len(self.switch_at) - 1):
            if self.switch_at[j] + self.transitions[j] > self.switch_at[j + 1]:
                raise ValueError("transitions overlap")

    def distribution(self, t: int) -> np.ndarray:
        i = int(np.searchsorted(self.switch_at, t, side="right"))
        if i == 0:
            return self.strategies[0]
        start, width = self.switch_at[i - 1], self.transitions[i - 1]
        if t >= start + width:
            return self.strategies[i]
        w = (t - start + 1) / (width + 1)
        return (1 - w) * self.strategies[i - 1] + w * self.strategies[i]

    def act(self, t: int, rng: np.random.Generator) -> int:
        return int(rng.choice(N_ACTIONS, p=self.distribution(t)))


def _sample_strategy(rng: np.random.Generator) -> np.ndarray:
    while True:
        p = rng.dirichlet(np.ones(N_ACTIONS))
        if not near_held_out(p):
            return p


def sample_pretraining_opponent(rng: np.random.Generator, n_rounds: int = 600) -> ScheduledOpponent:
    n_switches = int(rng.integers(0, MAX_SWITCHES + 1))
    while True:
        transitions = [0 if rng.random() < 0.5 else int(rng.integers(1, MAX_TRANSITION + 1)) for _ in range(n_switches)]
        starts = sorted(int(s) for s in rng.integers(EDGE_MARGIN, n_rounds - EDGE_MARGIN, size=n_switches))
        ends = [s + w for s, w in zip(starts, transitions)]
        if all(ends[j] + MIN_GAP <= starts[j + 1] for j in range(n_switches - 1)) and (
            not ends or ends[-1] <= n_rounds - EDGE_MARGIN
        ):
            break

    strategies = [_sample_strategy(rng)]
    for w in transitions:
        while True:
            q = _sample_strategy(rng)
            if tv(strategies[-1], q) >= MIN_SWITCH_TV and not (w > 0 and segment_near_held_out(strategies[-1], q)):
                break
        strategies.append(q)
    return ScheduledOpponent(strategies, starts, transitions)
