"""The two exploratory agents: online fine-tuning and change-aware RL.

See AGENT_SPECS.md, sections 2 and 3, and the tuning-protocol amendment.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from regime.agents.base import Agent
from regime.env import N_ACTIONS


class FineTuneAgent(Agent):
    """MLP policy over the last k opponent moves, one REINFORCE step per round.

    Implemented in numpy (forward, backward and Adam by hand): at batch size
    1 on a 32-unit network, PyTorch's per-op overhead made each episode
    ~100x slower. Initialization and Adam match PyTorch's defaults
    (tests/test_online_agents.py checks the gradient step against torch).
    """

    name = "fine_tune"
    BETAS, EPS = (0.9, 0.999), 1e-8

    def __init__(self, lr: float = 1e-2, baseline_decay: float = 0.9, window: int = 10, hidden: int = 32):
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.window = window
        self.hidden = hidden

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        n_in = N_ACTIONS * self.window

        def linear(fan_in, fan_out):  # torch.nn.Linear default init
            bound = 1 / np.sqrt(fan_in)
            return rng.uniform(-bound, bound, (fan_out, fan_in)), rng.uniform(-bound, bound, fan_out)

        W1, b1 = linear(n_in, self.hidden)
        W2, b2 = linear(self.hidden, N_ACTIONS)
        self.params = [W1, b1, W2, b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.history: deque[int] = deque(maxlen=self.window)
        self.baseline = 0.0
        self._forward()

    def _input(self) -> np.ndarray:
        x = np.zeros((self.window, N_ACTIONS))
        # Most recent move in the last slot; earlier slots stay zero until the window fills.
        for i, m in enumerate(self.history):
            x[self.window - len(self.history) + i, m] = 1.0
        return x.ravel()

    def _forward(self) -> None:
        W1, b1, W2, b2 = self.params
        self._x = self._input()
        self._h = np.tanh(W1 @ self._x + b1)
        self._logits = W2 @ self._h + b2

    def output_scores(self) -> np.ndarray:
        return self._logits.copy()

    def policy(self) -> np.ndarray:
        z = np.exp(self._logits - self._logits.max())
        return z / z.sum()

    def gradients(self, action: int, advantage: float) -> list[np.ndarray]:
        """Gradients of the loss -advantage * log pi(action | x)."""
        W1, b1, W2, b2 = self.params
        dlogits = -advantage * (np.eye(N_ACTIONS)[action] - self.policy())
        dh = W2.T @ dlogits * (1 - self._h**2)
        return [np.outer(dh, self._x), dh, np.outer(dlogits, self._h), dlogits]

    def observe(self, my_action: int, opp_action: int, reward: float) -> None:
        # Policy-gradient step on the input the action was chosen from.
        grads = self.gradients(my_action, reward - self.baseline)
        self.t += 1
        b1c, b2c = 1 - self.BETAS[0] ** self.t, 1 - self.BETAS[1] ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.BETAS[0]
            m += (1 - self.BETAS[0]) * g
            v *= self.BETAS[1]
            v += (1 - self.BETAS[1]) * g**2
            p -= self.lr * (m / b1c) / (np.sqrt(v / b2c) + self.EPS)
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        self.history.append(opp_action)
        self._forward()

    def internal_state(self) -> np.ndarray:
        return self._h.copy()

    def weight_vector(self) -> np.ndarray:
        """Flattened weights, for the exploratory weight-displacement signal."""
        return np.concatenate([p.ravel() for p in self.params])


class ChangeAwareAgent(Agent):
    """Recency-weighted action values with surprise-scaled exploration."""

    name = "change_aware"

    def __init__(self, forgetting: float = 0.9, base_temperature: float = 0.1, surprise_gain: float = 1.0):
        self.forgetting = forgetting
        self.base_temperature = base_temperature
        self.surprise_gain = surprise_gain

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        self.q = np.zeros(N_ACTIONS)
        self.fast = self.slow = 0.0
        self.n = 0

    def temperature(self) -> float:
        ratio = self.fast / self.slow if self.slow > 0 else 1.0
        return self.base_temperature * (1 + self.surprise_gain * max(0.0, ratio - 1))

    def output_scores(self) -> np.ndarray:
        return self.q.copy()

    def policy(self) -> np.ndarray:
        z = self.q / self.temperature()
        z = np.exp(z - z.max())
        return z / z.sum()

    def observe(self, my_action: int, opp_action: int, reward: float) -> None:
        rate = 1 - self.forgetting
        surprise = abs(reward - self.q[my_action])
        if self.n == 0:
            self.fast = self.slow = surprise
        else:
            self.fast += rate * (surprise - self.fast)
            self.slow += rate / 10 * (surprise - self.slow)
        self.q[my_action] += rate * (reward - self.q[my_action])
        self.n += 1

    def internal_state(self) -> np.ndarray:
        # Contains the output scores (Q): lead/lag is ranking-only for this agent (AGENT_SPECS.md).
        return np.concatenate([self.q, [self.fast, self.slow, self.temperature()]])
