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
# Layer identity in the lag figures: layer 1 = the in-context blue, layer 2 (structural null) = slot 5
# magenta (below 3:1 on light, so every mark carries a visible label). Corrected = filled, raw = hollow.
LAYER_COLOR = {1: "#2a78d6", 2: "#e87ba4"}
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
    l2 = load("results/exploratory/structural_null_layer2_lags.json")
    l2s = load("results/exploratory/structural_null_layer2.json")
    assert l2["layer1_corrected"] == ic["test_b"]["lags"]
    rows = [
        (1, "Layer 1 (tested)", [("corrected (pre-registered)", ic["test_b"]), ("raw — exploratory", ic["exploratory_raw_state_lag"])]),
        (2, "Layer 2 (structural null)",
         [("corrected", dict(l2s["layer2_corrected"], lags=l2["layer2_corrected"])),
          ("raw", dict(l2s["layer2_raw"], lags=l2["layer2_raw"]))]),
    ]
    all_lags = [x for _, _, panels in rows for _, r in panels for x in r["lags"]]
    bins = np.arange(min(all_lags) - 0.5, max(all_lags) + 1.5)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), sharex=True, sharey=True)
    for (layer, row_title, panels), row_axes in zip(rows, axes):
        color = LAYER_COLOR[layer]
        for j, (ax, (label, res)) in enumerate(zip(row_axes, panels)):
            raw = j == 1
            ax.hist(res["lags"], bins=bins, color=SURFACE if raw else color, edgecolor=color, linewidth=1)
            c_lo, c_hi = res["median_ci95"]
            ax.axvspan(c_lo, c_hi, color=INK_2, alpha=0.12, linewidth=0)
            ax.axvline(res["median_lag"], color=INK, linewidth=1.5)
            ax.axvline(0, color=INK_2, linewidth=1, linestyle=":")
            ax.set_title(f"{row_title}: {label}", fontsize=10)
            ax.text(0.02, 0.95, f"median {res['median_lag']:+g}  [{c_lo:g}, {c_hi:g}]\n"
                    f"{res['n_positive']} + / {res['n_negative']} − / {res['n_zero']} zero",
                    transform=ax.transAxes, va="top", fontsize=9, color=INK_2)
        row_axes[0].set_ylabel("seeds")
    for ax in axes[1]:
        ax.set_xlabel("lag (rounds; + = state moves first)")
    fig.suptitle("Per-seed lags against the output: layer 1 vs layer 2, a structural null (no information lag possible)",
                 fontweight="semibold", color=INK, fontsize=11)
    fig.tight_layout()
    savefig(fig, "3_lag_histograms.png")


def fig_robustness(conf: dict) -> None:
    ic = conf["agents"]["in_context"]
    ex = load("results/exploratory/2_4_5_in_context.json")
    un = load("results/exploratory/3_untrained.json")
    l2 = load("results/exploratory/structural_null_layer2.json")
    # (label, result, layer color or None for gray, hollow marker for raw)
    rows = [
        ("Layer 1, corrected (confirmatory)", ic["test_b"], LAYER_COLOR[1], False),
        ("Layer 2, corrected — structural null", l2["layer2_corrected"], LAYER_COLOR[2], False),
        ("Layer 1, raw", ic["exploratory_raw_state_lag"], LAYER_COLOR[1], True),
        ("Layer 2, raw — structural null", l2["layer2_raw"], LAYER_COLOR[2], True),
        ("Behavior = logits (no saturation)", ex["2_logit_behavior"], None, False),
        ("h = 7", ex["4_h7"], None, False),
        ("h = 28", ex["4_h28"], None, False),
        ("Onset difference (threshold-based)", ex["5_onset"], None, False),
        ("Untrained network, corrected state", un["3_untrained_corrected"], None, False),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (label, r, color, hollow) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = r["median_ci95"]
        c = color or INK_2
        ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(r["median_lag"], y, "o", markersize=8, color=c, markerfacecolor=SURFACE if hollow else c,
                markeredgecolor=c if hollow else SURFACE, markeredgewidth=2)
        n_nz = r["n_positive"] + r["n_negative"]
        if not n_nz:
            note = "all zero"
        elif r["median_lag"] > 0:
            note = f"{r['n_positive']}/{n_nz} positive"
        else:
            note = f"{r['n_negative']}/{n_nz} negative"
        ax.text(max(hi, r["median_lag"]) + 0.6, y, note, va="center", fontsize=8.5, color=INK_2)
    ax.axhline(len(rows) - 4.5, color=GRID, linewidth=1)
    ax.axvline(0, color=INK_2, linewidth=1, linestyle=":")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set(xlabel="median lag, 95% CI (rounds; + = state moves first)",
           title="Lag against the output — layer 2 is a structural null; all rows but the first are exploratory")
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


def fig_structural_nulls() -> None:
    """How the measurement artifact was caught: the lag appears in almost every residual direction."""
    dim = load("results/exploratory/dimension_diagnostics.json")["d2"]["by_k"]
    null = load("results/exploratory/structural_null_layer2.json")
    ln = load("results/exploratory/layernorm_diagnostic.json")
    ks = [int(k) for k in dim]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    rng = np.random.default_rng(0)
    for i, k in enumerate(ks):
        meds = np.array(dim[str(k)]["median_lags"])
        x = i + rng.uniform(-0.12, 0.12, len(meds))
        ax.plot(x, meds, "o", markersize=6, color=INK_2, alpha=0.55, markeredgewidth=0)
    refs = [("Layer 1, corrected (study 1)", null["layer1_corrected_reference"]["median_lag"], LAYER_COLOR[1], "-"),
            ("Layer 2, corrected (structural null)", null["layer2_corrected"]["median_lag"], LAYER_COLOR[2], "-"),
            ("Layer 1, LayerNorm-normalized", ln["layer1_layernorm"]["median_lag"], LAYER_COLOR[1], "--"),
            ("Layer 2, LayerNorm-normalized", ln["layer2_layernorm"]["median_lag"], LAYER_COLOR[2], "--")]
    for label, y, color, style in refs:
        ax.axhline(y, color=color, linewidth=1.5, linestyle=style, label=f"{label}: {y:+g}")
    ax.axhline(0, color=INK_2, linewidth=1, linestyle=":")
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set(xlabel="dimension k of a random orthonormal projection of the layer-1 state (20 projections each)",
           ylabel="median lag vs output (rounds)",
           title="Structural nulls — the lag appears in almost every residual direction (exploratory)")
    ax.set_ylim(-11.5, 11.5)
    ax.legend(loc="upper right", fontsize=8.5)
    savefig(fig, "6_structural_nulls.png")


def fig_study3() -> None:
    r = load("results/study3/results.json")
    order = ["fine_tune", "change_aware", "in_context"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (title, vals, errs) in zip(axes, [
        ("Score vs a reactive opponent (Nash = 0)", [r["H3"][a]["mean"] for a in order],
         [[r["H3"][a]["mean"] - r["H3"][a]["ci95"][0] for a in order], [r["H3"][a]["ci95"][1] - r["H3"][a]["mean"] for a in order]]),
        ("Predictability: trailing-5 entropy (nats)", [r["H1"]["entropy"][a] for a in order], None),
    ]):
        x = np.arange(len(order))
        ax.bar(x, vals, width=0.6, color=[AGENT_COLOR[a] for a in order], edgecolor=SURFACE, linewidth=2)
        if errs:
            ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3)
            ax.axhline(0, color=INK_2, linewidth=1)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:+.3f}" if errs else f"{v:.3f}", (xi, v), xytext=(0, 4 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=9, color=INK_2)
        ax.set_xticks(x, [AGENT_LABEL[a] for a in order])
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(-0.2, 0.28)
    axes[1].set_ylim(0, 1.25)
    axes[1].axhline(np.log(3), color=INK_2, linewidth=1, linestyle=":")
    axes[1].text(-0.35, np.log(3) + 0.02, "uniform play (ln 3)", va="bottom", ha="left", fontsize=8.5, color=INK_2)
    fig.suptitle("Study 3: frozen agents against a fictitious-play opponent (400 seeds each)",
                 fontweight="semibold", color=INK, fontsize=11)
    fig.tight_layout()
    savefig(fig, "7_study3.png")


def fig_study3_robustness_and_mscale() -> None:
    """Study 3b (confirmatory) and M-scaling (exploratory), labeled as such in each panel's title."""
    s3r = load("results/study3/results.json")["H3"]
    lag = load("results/study3/obs_lag.json")["conditions"]
    noise = load("results/study3/obs_noise.json")["conditions"]
    ms = load("results/study3/mscale.json")["agents"]
    mi = load("results/study3/mscale_in_context.json")["by_M"]
    order = ["fine_tune", "change_aware", "in_context"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))

    conds = [("none\n(study 3)", None), ("lag,\nsymmetric", lag["symmetric"]), ("lag,\nagent only", lag["agent_only"]),
             ("noise,\nsymmetric", noise["symmetric"]), ("noise,\nagent only", noise["agent_only"])]
    x = np.arange(len(conds))
    # Categorical conditions: dots, not lines (no continuum between them)
    for k, ag in enumerate(order):
        ys = [s3r[ag]["mean"] if c is None else c["agents"][ag]["score"] for _, c in conds]
        a.plot(x + (k - 1) * 0.12, ys, "o", color=AGENT_COLOR[ag], markersize=8, markeredgecolor=SURFACE,
               markeredgewidth=1.5)
        a.annotate(AGENT_LABEL[ag], (x[0] - 0.12, ys[0]), xytext=(-6, 0), textcoords="offset points",
                   ha="right", va="center", fontsize=8.5, color=INK_2)
    a.axhline(0, color=INK_2, linewidth=1, linestyle=":")
    a.set_xticks(x, [c for c, _ in conds], fontsize=8.5)
    a.set_xlim(-1.4, len(conds) - 0.5)
    a.grid(axis="x", visible=False)
    a.set(ylabel="score (Nash = 0)", title="Latency and noise (study 3b) — confirmatory")

    Ms = [5, 10, 20, 50]
    series = {"fine_tune": [s3r["fine_tune"]["mean"]] + [ms["fine_tune"]["by_M"][str(m)]["score"] for m in Ms[1:]],
              "change_aware": [s3r["change_aware"]["mean"]] + [ms["change_aware"]["by_M"][str(m)]["score"] for m in Ms[1:]],
              "in_context": [s3r["in_context"]["mean"]] + [mi[str(m)]["score"] for m in Ms[1:]]}
    for ag in order:
        b.plot(Ms, series[ag], "-o", color=AGENT_COLOR[ag], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        b.annotate(AGENT_LABEL[ag], (Ms[-1], series[ag][-1]), xytext=(6, 0), textcoords="offset points",
                   va="center", fontsize=8.5, color=INK_2)
    b.axhline(0, color=INK_2, linewidth=1, linestyle=":")
    b.set_xscale("log")
    b.set_xticks(Ms, [str(m) for m in Ms])
    b.minorticks_off()
    b.set(xlabel="opponent window M (larger = slower opponent)", ylabel="score (Nash = 0)",
          title="Opponent speed (M-scaling) — exploratory")
    b.set_xlim(4, 90)
    fig.suptitle("Study 3: frozen agents vs a reactive opponent (400 seeds per point)",
                 fontweight="semibold", color=INK, fontsize=11)
    fig.tight_layout()
    savefig(fig, "8_study3_robustness_mscale.png")


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
    fig_structural_nulls()
    fig_study3()
    fig_study3_robustness_and_mscale()


if __name__ == "__main__":
    main()
