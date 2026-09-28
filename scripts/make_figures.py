"""Figures for the README, from the frozen confirmatory and exploratory results.

    python scripts/make_figures.py

Regret curves and the drift overlay need per-round data the results files
don't store, so they come from deterministic reruns of the analysis seeds.
The script asserts the rerun reproduces the confirmatory lags before plotting.
Writes figures/*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from regime.analysis import AGENTS, ANALYSIS_SWITCH_SEEDS, lag_estimate, run_many, signals
from regime.metrics import moving_average
from scripts_common import load_cal

OUT = Path("figures")
# Reference categorical palette, slots 1-3 (validated all-pairs, light mode).
AGENT_COLOR = {"in_context": "#2a78d6", "change_aware": "#eb6834", "fine_tune": "#1baf7a"}
AGENT_LABEL = {"in_context": "In-context", "change_aware": "Change-aware RL", "fine_tune": "Fine-tuning"}
SIGNAL_COLOR = {"state": "#4a3aa7", "output": "#e34948"}  # slots 7 and 8: not agent identities
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "semibold", "lines.linewidth": 2, "legend.frameon": False,
})


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def savefig(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def mean_band(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rows.mean(0), 1.96 * rows.std(0) / np.sqrt(len(rows))


def dodge(ys: dict, gap: float) -> dict:
    """Spread label positions so neighbors are at least `gap` apart."""
    items = sorted(ys.items(), key=lambda kv: kv[1])
    out, last = {}, -np.inf
    for k, y in items:
        out[k] = max(y, last + gap)
        last = out[k]
    return out


def fig_regret(logs: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ends = {}
    for agent in AGENTS:
        m, b = mean_band(np.stack([moving_average(l.instant_regret, 20) for l in logs[agent]]))
        t = np.arange(len(m))
        ax.fill_between(t, m - b, m + b, color=AGENT_COLOR[agent], alpha=0.15, linewidth=0)
        ax.plot(t, m, color=AGENT_COLOR[agent], label=AGENT_LABEL[agent])
        ends[agent] = m[-1]
    for agent, y in dodge(ends, 0.035).items():
        ax.annotate(AGENT_LABEL[agent], (len(m) - 1, y), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.axvline(200, color=INK_2, linewidth=1, linestyle="--")
    ax.text(203, ax.get_ylim()[1] * 0.95, "switch A → B", color=INK_2, fontsize=9, va="top")
    ax.set(xlabel="round", ylabel="regret per round (20-round window)",
           title="Regret around the switch (A → B, 100 seeds, mean and 95% band)")
    ax.legend(loc="upper right")
    savefig(fig, "1_regret_curves.png")


def fig_overlay(ic_logs: list, cal) -> None:
    a, W = cal.switch_at, cal.W
    win = slice(a - W, a + W + 1)

    def z(x):
        return (x - x.mean()) / x.std()

    ds, do = zip(*[signals(l, cal) for l in ic_logs])
    rel = np.arange(-W, W + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, series in [("state", ds), ("output", do)]:
        m, b = mean_band(np.stack([z(s[win]) for s in series]))
        ax.fill_between(rel, m - b, m + b, color=SIGNAL_COLOR[name], alpha=0.15, linewidth=0)
        label = "Layer-1 state (corrected)" if name == "state" else "Output scores"
        ax.plot(rel, m, color=SIGNAL_COLOR[name], label=label)
    ax.axvline(0, color=INK_2, linewidth=1, linestyle="--")
    ax.set(xlabel="rounds relative to the switch", ylabel=f"displacement over h = {cal.h} rounds (z-scored per seed)",
           title="In-context agent: state vs output displacement (confirmatory window)")
    ax.legend(loc="upper right")
    savefig(fig, "2_drift_overlay.png")


def fig_lags(conf: dict) -> None:
    ic = conf["agents"]["in_context"]
    panels = [("Corrected state (pre-registered)", ic["test_b"]),
              ("Raw state — exploratory", ic["exploratory_raw_state_lag"])]
    lo = min(min(p["lags"]) for _, p in panels)
    hi = max(max(p["lags"]) for _, p in panels)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, res) in zip(axes, panels):
        lags = np.array(res["lags"])
        bins = np.arange(lo - 0.5, hi + 1.5)
        ax.hist(lags, bins=bins, color=AGENT_COLOR["in_context"], edgecolor=SURFACE, linewidth=1)
        c_lo, c_hi = res["median_ci95"]
        ax.axvspan(c_lo, c_hi, color=INK_2, alpha=0.12, linewidth=0)
        ax.axvline(res["median_lag"], color=INK, linewidth=1.5)
        ax.axvline(0, color=INK_2, linewidth=1, linestyle=":")
        ax.set(title=title, xlabel="lag (rounds; + = state moves first)")
        ax.text(0.02, 0.95, f"median {res['median_lag']:+g}  [{c_lo:g}, {c_hi:g}]\n"
                f"{res['n_positive']} + / {res['n_negative']} − / {res['n_zero']} zero",
                transform=ax.transAxes, va="top", fontsize=9, color=INK_2)
    axes[0].set_ylabel("seeds")
    savefig(fig, "3_lag_histograms.png")


def fig_robustness(conf: dict) -> None:
    ic = conf["agents"]["in_context"]
    ex = load("results/exploratory/2_4_5_in_context.json")
    un = load("results/exploratory/3_untrained.json")
    rows = [
        ("Confirmatory: corrected state, h = 14", ic["test_b"]),
        ("Behavior = logits (no saturation)", ex["2_logit_behavior"]),
        ("h = 7", ex["4_h7"]),
        ("h = 28", ex["4_h28"]),
        ("Onset difference (threshold-based)", ex["5_onset"]),
        ("Raw state (no latest-move correction)", ic["exploratory_raw_state_lag"]),
        ("Untrained network, corrected state", un["3_untrained_corrected"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (label, r) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = r["median_ci95"]
        color = AGENT_COLOR["in_context"] if i == 0 else INK_2
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(r["median_lag"], y, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        n_nz = r["n_positive"] + r["n_negative"]
        if not n_nz:
            note = "all zero"
        elif r["median_lag"] > 0:
            note = f"{r['n_positive']}/{n_nz} positive"
        else:
            note = f"{r['n_negative']}/{n_nz} negative"
        ax.text(max(hi, r["median_lag"]) + 0.6, y, note, va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1, linestyle=":")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set(xlabel="median lag, 95% CI (rounds; + = state moves first)",
           title="In-context lag across checks — first row confirmatory, the rest exploratory")
    ax.set_xlim(-7, 11)
    savefig(fig, "4_robustness.png")


def fig_ranking(conf: dict) -> None:
    ex = load("results/exploratory/6_7_8.json")
    order = ["in_context", "change_aware", "fine_tune"]
    panels = [
        ("Excess regret (pre-registered ranking)",
         [conf["ranking"][a]["headline_common_switch_200"]["excess_regret_mean"] for a in order],
         [conf["ranking"][a]["headline_common_switch_200"]["excess_regret_se"] for a in order]),
        ("Total regret over 600 rounds — exploratory",
         [ex[a]["6_total_regret_A_to_B"][0] for a in order], [ex[a]["6_total_regret_A_to_B"][1] for a in order]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (title, means, ses) in zip(axes, panels):
        x = np.arange(len(order))
        ax.bar(x, means, width=0.6, color=[AGENT_COLOR[a] for a in order], edgecolor=SURFACE, linewidth=2)
        err = 1.96 * np.array(ses)
        ax.errorbar(x, means, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3)
        for xi, m, e in zip(x, means, err):
            ax.annotate(f"{m:.1f}", (xi, m + e), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=INK_2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(np.array(means) + err) * 1.12)
        ax.set_xticks(x, [AGENT_LABEL[a] for a in order])
        ax.set(title=title, ylabel="regret (lower is better)")
        ax.grid(axis="x", visible=False)
    savefig(fig, "5_ranking.png")


def main() -> None:
    conf = load("results/confirmatory/results.json")
    logs = {a: run_many(a, "switch", ANALYSIS_SWITCH_SEEDS, 200, 600) for a in AGENTS}

    cal = load_cal("in_context")
    assert (cal.switch_at, cal.n_rounds) == (200, 600)  # its own switch is the common setup
    repro = [lag_estimate(*signals(l, cal), cal.switch_at, cal.W, cal.L) for l in logs["in_context"]]
    assert repro == conf["agents"]["in_context"]["test_b"]["lags"], "rerun does not reproduce confirmatory lags"
    print("reproduction check passed")

    fig_regret(logs)
    fig_overlay(logs["in_context"], cal)
    fig_lags(conf)
    fig_robustness(conf)
    fig_ranking(conf)


if __name__ == "__main__":
    main()
