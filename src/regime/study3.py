"""Study 3: a reactive (fictitious-play) opponent. Rules: ANALYSIS_PLAN.md, "Study 3"."""

from __future__ import annotations

import numpy as np

N_ROUNDS = 1000
EPS = 0.1
M_GRID = (5, 10, 20, 50)
M_FALLBACK = 20
SELECTION_WINDOW = (300, 1000)
M_SEEDS = range(50000, 50100)
TW_SEEDS = range(50100, 50200)
TEST_SEED_BASE = 60000
DRY_RUN = dict(window=7, eps=0.5, seeds=range(69000, 69020))
SESOI = 0.02
N_GRID = (100, 200, 400, 800)
INFLATION = max(1.0, 22.9 / 23.2)  # study 1: most variable frozen agent's SD / most variable reference's


def reference_agents() -> dict:
    from regime.agents import ConstantAgent, FrequencyAgent, UniformAgent
    from regime.env import ROCK

    return {"uniform": UniformAgent, "always_rock": lambda: ConstantAgent(ROCK),
            "freq_w20": lambda: FrequencyAgent(window=20), "freq_w50": lambda: FrequencyAgent(window=50),
            "freq_all": lambda: FrequencyAgent()}


def score(expected_rewards: np.ndarray, start: int, stop: int | None = None) -> float:
    return float(np.mean(expected_rewards[start:stop]))


def separated(means: dict, ses: dict) -> bool:
    """Every pair separated by more than 2 SE of the difference (strict >)."""
    keys = list(means)
    return all(abs(means[a] - means[b]) > 2 * np.hypot(ses[a], ses[b])
               for i, a in enumerate(keys) for b in keys[i + 1 :])


def trailing_mean(x: np.ndarray, k: int = 50) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - k)
    return (c[i] - c[lo]) / (i - lo)


def settle_round(m: np.ndarray, final: tuple[int, int] = (800, 1000), frac: float = 0.1, floor: float = 0.01) -> int:
    """First round after which |m(t) - m_final| <= max(frac * |m(0) - m_final|, floor) for all later t (closed <=)."""
    m_final = float(m[final[0] : final[1]].mean())
    band = max(frac * abs(m[0] - m_final), floor)
    outside = np.flatnonzero(np.abs(m - m_final) > band)
    return 0 if len(outside) == 0 else int(outside[-1] + 1)


def required_n(sigma: float, delta: float = SESOI, alpha: float = 0.05 / 3, power: float = 0.8) -> float:
    """Per-group n for a two-sided two-sample comparison of means (normal approximation)."""
    from scipy.stats import norm

    return 2 * ((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sigma / delta) ** 2


def choose_n(sigma: float) -> tuple[int, bool]:
    need = required_n(sigma * INFLATION)
    for n in N_GRID:
        if n >= need:
            return n, False
    return N_GRID[-1], True


def trailing_entropy(actions: np.ndarray, window: int) -> np.ndarray:
    """Entropy (nats) of the empirical distribution of the last `window` actions, per round."""
    onehot = np.eye(3)[actions]
    c = np.cumsum(np.vstack([np.zeros(3), onehot]), axis=0)
    i = np.arange(1, len(actions) + 1)
    counts = c[i] - c[np.maximum(0, i - window)]
    p = counts / counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)


def cycle_period(br_actions: np.ndarray, lo: int = 2, hi: int = 200, min_peak: float = 0.2) -> int | None:
    """Lag of the first local maximum above `min_peak` in the autocorrelation of a one-hot action
    sequence (averaged over actions), searched over [lo, hi]; None if there is none."""
    x = np.eye(3)[br_actions] - np.eye(3)[br_actions].mean(0)
    den = (x * x).sum()
    if den == 0:
        return None
    ac = np.array([(x[:-k] * x[k:]).sum() / den for k in range(1, hi + 2)])
    for k in range(lo, hi + 1):
        if ac[k - 1] > min_peak and ac[k - 1] >= ac[k - 2] and ac[k - 1] >= ac[k]:
            return k
    return None


BRIDGE = dict(switch_at=200, seeds=range(61000, 61400), scripted=(50, 200), early=(200, 400))


class BridgeOpponent:
    """Bridge condition (b): study 1's Strategy A for `switch_at` rounds, then fictitious play.
    The fictitious-play component observes the agent from round 0."""

    def __init__(self, window: int, eps: float, switch_at: int = 200):
        from regime.env import STRATEGY_A, FictitiousPlayOpponent

        self.scripted = np.asarray(STRATEGY_A, dtype=float)
        self.fp = FictitiousPlayOpponent(window, eps)
        self.switch_at = self.switch_end = switch_at

    def reset(self) -> None:
        self.fp.reset()

    def observe_agent(self, action: int) -> None:
        self.fp.observe_agent(action)

    def distribution(self, t: int) -> np.ndarray:
        return self.scripted if t < self.switch_at else self.fp.distribution(t)

    def act(self, t: int, rng: np.random.Generator) -> int:
        return int(rng.choice(3, p=self.distribution(t)))
