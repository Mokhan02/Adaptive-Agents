"""Study 3 (ANALYSIS_PLAN.md, "Study 3").

    python scripts/study3.py calibrate   # reference agents only; writes results/study3/calibration.json
    python scripts/study3.py run --dry-run  # different opponent (M=7, eps=0.5), 20 seeds; not results
    python scripts/study3.py run            # once, from the tag study3-v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from regime import study3 as s3

OUT = Path("results/study3")


def _ref_run(name: str, window: int, n_rounds: int, seed: int) -> np.ndarray:
    from regime.env import FictitiousPlayOpponent
    from regime.runner import run_episode

    agent = s3.reference_agents()[name]()
    return run_episode(agent, FictitiousPlayOpponent(window, s3.EPS), n_rounds, seed).expected_rewards


def ref_runs(name: str, window: int, seeds, n_rounds: int = s3.N_ROUNDS) -> np.ndarray:
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        return np.stack(list(pool.map(partial(_ref_run, name, window, n_rounds), seeds, chunksize=5)))


def calibrate() -> None:
    out = OUT / "calibration.json"
    if out.exists():
        raise SystemExit(f"{out} exists; calibration runs once")
    names = list(s3.reference_agents())
    lo, hi = s3.SELECTION_WINDOW

    selection, chosen = {}, None
    for M in s3.M_GRID:
        scores = {n: ref_runs(n, M, s3.M_SEEDS)[:, lo:hi].mean(1) for n in names}
        means = {n: float(v.mean()) for n, v in scores.items()}
        ses = {n: float(v.std() / np.sqrt(len(v))) for n, v in scores.items()}
        ok = s3.separated(means, ses)
        selection[M] = dict(means=means, ses=ses, all_pairs_separated=ok)
        print(f"M={M:>2}: " + "  ".join(f"{n} {means[n]:+.3f}±{ses[n]:.3f}" for n in names) + f"  separated={ok}", flush=True)
        if ok and chosen is None:
            chosen = M
            break
    rule_failed = chosen is None
    M = s3.M_FALLBACK if rule_failed else chosen

    tw_runs = {n: ref_runs(n, M, s3.TW_SEEDS) for n in names}
    tw = {n: s3.settle_round(np.mean([s3.trailing_mean(r) for r in runs], axis=0)) for n, runs in tw_runs.items()}
    T_w = max(tw.values())
    n_rounds = s3.N_ROUNDS if T_w <= 700 else T_w + 300
    sigmas = {n: float(runs[:, T_w:].mean(1).std()) for n, runs in tw_runs.items()}
    sigma_ref = max(sigmas.values())
    N, underpowered = s3.choose_n(sigma_ref)
    power_table = {n: float(s3.required_n(sigma_ref * s3.INFLATION)) for n in ["required_n_per_agent"]}

    result = dict(
        M=M, M_rule_failed=rule_failed, selection=selection, eps=s3.EPS,
        T_w=T_w, T_w_by_reference=tw, n_rounds=n_rounds,
        score_sd_by_reference=sigmas, sigma_ref=sigma_ref, inflation=s3.INFLATION,
        sesoi=s3.SESOI, N=N, underpowered=underpowered, **power_table,
        seeds=dict(M_selection=[s3.M_SEEDS.start, s3.M_SEEDS.stop - 1], T_w=[s3.TW_SEEDS.start, s3.TW_SEEDS.stop - 1]),
        note="reference agents only; the three frozen agents were not run",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"M={M} (rule failed: {rule_failed})  T_w={T_w} {tw}  n_rounds={n_rounds}  sigma_ref={sigma_ref:.4f}  "
          f"required n={result['required_n_per_agent']:.0f} -> N={N}{' (underpowered)' if underpowered else ''}")


TAG = "study3-v1"
AGENTS = ("in_context", "change_aware", "fine_tune")
STUDY1_ORDERS = {  # best first
    "study1_preregistered_excess_regret": ["change_aware", "in_context", "fine_tune"],
    "study1_exploratory_total_regret": ["in_context", "change_aware", "fine_tune"],
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def preflight(out: Path) -> str:
    if out.exists():
        raise SystemExit(f"{out} exists: study 3 runs once")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes")
    head = git("rev-parse", "HEAD")
    if TAG not in git("tag", "--points-at", head).split():
        raise SystemExit(f"HEAD is not tagged {TAG}")
    return head


def _agent_run(name: str, window: int, eps: float, n_rounds: int, seed: int) -> dict:
    import torch

    from regime.analysis import make_agent
    from regime.env import FictitiousPlayOpponent
    from regime.runner import run_episode

    torch.set_num_threads(1)
    log = run_episode(make_agent(name), FictitiousPlayOpponent(window, eps), n_rounds, seed)
    return dict(expected=log.expected_rewards, actions=log.agent_actions, opp=log.opp_actions)


def replay_best_responses(actions: np.ndarray, window: int) -> np.ndarray:
    """The opponent's best-response action each round, replayed from the agent's moves (first index on
    ties; -1 before `window` moves). Diagnostic only."""
    from regime.env import expected_payoffs

    br = np.full(len(actions), -1)
    for t in range(window, len(actions)):
        p = np.bincount(actions[t - window : t], minlength=3) / window
        br[t] = int(np.argmax(expected_payoffs(p)))
    return br


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x))).mean(1)
    return boots, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def pair_outcome(a: str, b: str, diff: float, ci, p_holm: float) -> str:
    """H2 outcome for diff = score(a) - score(b), naming which agent is above."""
    lo, hi = ci
    if p_holm < 0.05:
        top, bottom = (a, b) if diff > 0 else (b, a)
        size = "by at least the effect of interest" if abs(diff) >= s3.SESOI else "by less than the effect of interest"
        return f"{top} above {bottom} {size}"
    if -s3.SESOI < lo and hi < s3.SESOI:
        return "equivalent within the margin"
    return "inconclusive"


def run(dry_run: bool) -> None:
    from scipy.stats import norm

    from regime.early_detection import holm

    cal = json.loads((OUT / "calibration.json").read_text())
    T_w, n_rounds, N = cal["T_w"], cal["n_rounds"], cal["N"]
    if dry_run:
        out = OUT / "dry_run" / "results.json"
        commit = git("rev-parse", "HEAD") + " (dry run)"
        window, eps, seeds = s3.DRY_RUN["window"], s3.DRY_RUN["eps"], s3.DRY_RUN["seeds"]
    else:
        out = OUT / "results.json"
        commit = preflight(out)
        window, eps, seeds = cal["M"], s3.EPS, range(s3.TEST_SEED_BASE, s3.TEST_SEED_BASE + N)
    progress = out.parent / "progress.log"
    out.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(progress, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        print(msg, flush=True)

    runs = {}
    for name in AGENTS:
        log(f"running {name}: {len(seeds)} seeds x {n_rounds} rounds (opponent M={window}, eps={eps})")
        with ProcessPoolExecutor(os.cpu_count()) as pool:
            runs[name] = list(pool.map(partial(_agent_run, name, window, eps, n_rounds), seeds, chunksize=5))
        log(f"done {name}")

    scores = {a: np.array([r["expected"][T_w:].mean() for r in runs[a]]) for a in AGENTS}
    entropy = {a: float(np.mean([s3.trailing_entropy(r["actions"], window)[T_w:].mean() for r in runs[a]])) for a in AGENTS}
    opp_entropy = {a: float(np.mean([s3.trailing_entropy(r["opp"], window)[T_w:].mean() for r in runs[a]])) for a in AGENTS}
    periods = {a: [s3.cycle_period(replay_best_responses(r["actions"], window)[T_w:]) for r in runs[a]] for a in AGENTS}

    # H3: each agent vs Nash (0), Holm across agents
    h3, p3 = {}, []
    for i, a in enumerate(AGENTS):
        boots, ci = bootstrap_mean_ci(scores[a], seed=i)
        p3.append(float(max(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())), 1e-4)))
        h3[a] = dict(mean=float(scores[a].mean()), ci95=ci)
    for a, ph in zip(AGENTS, holm(p3)):
        lo, hi = h3[a]["ci95"]
        h3[a]["p_holm"] = ph
        h3[a]["outcome"] = "exploited" if hi < 0 else "exploits the opponent" if lo > 0 else "not distinguishable from the Nash value"

    # H2: pairwise differences, Holm across the 3 pairs
    pairs = [(AGENTS[i], AGENTS[j]) for i in range(3) for j in range(i + 1, 3)]
    h2, p2 = {}, []
    rng = np.random.default_rng(1)
    for a, b in pairs:
        ba = rng.choice(scores[a], size=(10_000, len(scores[a]))).mean(1)
        bb = rng.choice(scores[b], size=(10_000, len(scores[b]))).mean(1)
        d = ba - bb
        diff = float(scores[a].mean() - scores[b].mean())
        p2.append(float(max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1e-4)))
        h2[f"{a} - {b}"] = dict(diff=diff, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    for (a, b), row, ph in zip(pairs, h2.values(), holm(p2)):
        row["p_holm"] = ph
        row["outcome"] = pair_outcome(a, b, row["diff"], row["ci95"], ph)

    # Ranking changes: a pair H2 separates (Holm p < 0.05) pointing against a study 1 order
    changes = {}
    for order_name, order in STUDY1_ORDERS.items():
        contradicted = []
        for (a, b), row in zip(pairs, h2.values()):
            if row["p_holm"] < 0.05:
                study3_a_better = row["diff"] > 0
                study1_a_better = order.index(a) < order.index(b)
                if study3_a_better != study1_a_better:
                    contradicted.append(f"{a} vs {b}")
        changes[order_name] = dict(changed=bool(contradicted), contradicted_pairs=contradicted)

    # H1: entropy ordering vs score ordering (ordinal check, n = 3)
    by_entropy = sorted(AGENTS, key=lambda a: entropy[a])
    by_score = sorted(AGENTS, key=lambda a: scores[a].mean())
    h1 = dict(entropy=entropy, order_by_entropy_low_to_high=by_entropy, order_by_score_low_to_high=by_score,
              orders_match=by_entropy == by_score,
              in_context_lowest_entropy_and_score=by_entropy[0] == "in_context" and by_score[0] == "in_context",
              status=("confirmed" if by_entropy == by_score and by_entropy[0] == "in_context" else "not confirmed")
              + " (ordinal check on n = 3, not a statistical test)")

    z = norm.ppf(1 - (0.05 / 3) / 2)
    achieved = {f"{a} - {b}": float(norm.cdf(s3.SESOI / np.sqrt(scores[a].var() / len(scores[a]) + scores[b].var() / len(scores[b])) - z))
                for a, b in pairs}
    results = dict(
        study="study 3: reactive (fictitious-play) opponent", commit=commit, tag=TAG, dry_run=dry_run,
        opponent=dict(M=window, eps=eps), T_w=T_w, n_rounds=n_rounds, n_per_agent=len(seeds),
        seeds=[seeds.start, seeds.stop - 1], score="mean expected reward over [T_w, n_rounds); Nash value 0",
        H1=h1, H2=h2, H3=h3, ranking_changes=changes, achieved_power_at_sesoi=achieved,
        diagnostics=dict(opponent_entropy=opp_entropy,
                         cycle_period_median={a: (float(np.median([p for p in v if p is not None])) if any(p is not None for p in v) else None)
                                              for a, v in periods.items()},
                         cycle_period_none_count={a: sum(p is None for p in v) for a, v in periods.items()}),
        example_trajectories={a: [dict(actions=runs[a][i]["actions"][T_w : T_w + 60].tolist(),
                                       opponent=runs[a][i]["opp"][T_w : T_w + 60].tolist()) for i in range(3)] for a in AGENTS},
    )
    out.write_text(json.dumps(results, indent=2))
    log(f"saved {out} (commit {commit})")
    for a in AGENTS:
        log(f"{a:<13} score {h3[a]['mean']:+.4f} {h3[a]['ci95']}  {h3[a]['outcome']}  entropy {entropy[a]:.3f}")
    for k, v in h2.items():
        log(f"{k:<28} diff {v['diff']:+.4f} {v['ci95']}  p_holm {v['p_holm']:.3g}  {v['outcome']}")
    log(f"H1: {h1['status']}  ranking changes: {changes}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["calibrate", "run"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.which == "run":
        run(args.dry_run)
    else:
        calibrate()


if __name__ == "__main__":
    main()
