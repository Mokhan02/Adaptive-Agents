"""Run the reference agents across seeds and plot regret around the switch.

    python scripts/sanity_check.py --seeds 20 --transition 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from regime.agents import FrequencyAgent, UniformAgent
from regime.env import STRATEGY_A, STRATEGY_B, RegimeSwitchOpponent
from regime.metrics import excess_regret, recovery_time, windowed_regret
from regime.runner import run_episode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=600)
    p.add_argument("--switch-at", type=int, default=200)
    p.add_argument("--transition", type=int, default=0)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("results/sanity"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    opponent = RegimeSwitchOpponent(STRATEGY_A, STRATEGY_B, args.switch_at, args.transition)
    agents = [UniformAgent, lambda: FrequencyAgent(), lambda: FrequencyAgent(window=20), lambda: FrequencyAgent(window=50)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    print(f"{'agent':<16}{'median recovery':>16}{'never':>10}{'excess regret':>15}")
    for make in agents:
        logs = [run_episode(make(), opponent, args.rounds, seed) for seed in range(args.seeds)]
        name = make().name
        curves = np.stack([windowed_regret(log) for log in logs])
        mean = curves.mean(0)
        se = curves.std(0) / np.sqrt(len(logs))
        (line,) = ax.plot(mean, label=name)
        ax.fill_between(np.arange(args.rounds), mean - se, mean + se, color=line.get_color(), alpha=0.2)

        excess = np.mean([excess_regret(log) for log in logs])
        try:
            rec = [recovery_time(log) for log in logs]
        except ValueError:
            print(f"{name:<16}{'n/a':>16}{'n/a':>10}{excess:>15.1f}")
            continue
        done = [r for r in rec if r is not None]
        med = f"{np.median(done):.0f}" if done else "-"
        print(f"{name:<16}{med:>16}{rec.count(None):>10}{excess:>15.1f}")

    ax.axvline(args.switch_at, color="k", ls="--", lw=1, label="switch")
    if args.transition:
        ax.axvspan(args.switch_at, opponent.switch_end, color="k", alpha=0.07)
    ax.set_xlabel("round")
    ax.set_ylabel("windowed regret (20 rounds)")
    ax.set_title(f"Reference agents, {args.seeds} seeds, transition={args.transition}")
    ax.legend()
    fig.tight_layout()
    path = args.out / f"regret_transition{args.transition}.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
