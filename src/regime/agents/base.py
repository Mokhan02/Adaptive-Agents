from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """An adaptive RPS player.

    Each round the runner calls `policy()` to get a distribution over actions,
    samples a move, then calls `observe()` with the outcome. After `observe()`,
    `internal_state()` and `output_scores()` are logged for drift analysis.
    """

    name: str = "agent"

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    @abstractmethod
    def policy(self) -> np.ndarray:
        """Distribution over actions for the current round."""

    def on_action(self, action: int) -> None:
        """Called when the agent's action for this round is sampled, before any feedback arrives.
        Agents whose update needs the decision-time input (fine-tuning, under lag) record it here."""

    @abstractmethod
    def observe(self, my_action: int, opp_action: int, reward: float) -> None:
        """Update after a round is played."""

    @abstractmethod
    def output_scores(self) -> np.ndarray:
        """Continuous pre-decision scores (e.g. logits) behind `policy()`.

        This is the "behavior" signal for lead/lag analysis. It must vary
        smoothly with the agent's evidence; a sampled or argmax action would
        make any continuous internal state appear to lead by construction.
        """

    def internal_state(self) -> np.ndarray | None:
        """Vector snapshot of the agent's internal state, or None if not tracked."""
        return None
