"""M-scaling addendum: the in-context agent (ANALYSIS_PLAN.md, "Study 3: M-scaling check", addendum).

    python scripts/mscale_icl.py     # after scripts/study3.py mscale has written results/study3/mscale.json
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from regime import study3 as s3
from regime.early_detection import holm
from study3 import OUT, _obs_agent_run, change_outcome, git

MS, SEEDS = (10, 20, 50), range(66000, 66400)


def main() -> None:
    out = OUT / "mscale_in_context.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes")
    commit = git("rev-parse", "HEAD")
    first = json.loads((OUT / "mscale.json").read_text())
    cal = json.loads((OUT / "calibration.json").read_text())
    T_w, n_rounds = cal["T_w"], cal["n_rounds"]
    base = {a: np.array(v) for a, v in json.loads((OUT / "baseline_rerun.json").read_text()).items()}
    progress = OUT / "mscale_progress.log"

    def run(agent, M):
        with open(progress, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} running {agent} M={M} (addendum)\n")
        with ProcessPoolExecutor(os.cpu_count()) as pool:
            runs = list(pool.map(partial(_obs_agent_run, agent, M, s3.EPS, n_rounds, {}), SEEDS, chunksize=5))
        return np.array([r["expected"][T_w:].mean() for r in runs])

    rl = {M: run("change_aware", M) for M in MS}
    for M in MS:
        assert float(rl[M].mean()) == first["agents"]["change_aware"]["by_M"][str(M)]["score"], "RL rerun differs"
    icl = {M: run("in_context", M) for M in MS}

    rng = np.random.default_rng(300)
    b5_i = rng.choice(base["in_context"], (10_000, 400)).mean(1)
    b5_r = rng.choice(base["change_aware"], (10_000, 400)).mean(1)
    rows, ps, did = {}, [], {}
    for M in MS:
        d_i = rng.choice(icl[M], (10_000, 400)).mean(1) - b5_i
        d_r = rng.choice(rl[M], (10_000, 400)).mean(1) - b5_r
        delta = float(icl[M].mean() - base["in_context"].mean())
        ps.append(float(max(min(1.0, 2 * min((d_i <= 0).mean(), (d_i >= 0).mean())), 1e-4)))
        rows[str(M)] = dict(score=float(icl[M].mean()), delta_vs_M5=delta,
                            ci95=[float(np.percentile(d_i, 2.5)), float(np.percentile(d_i, 97.5))])
        dd = d_i - d_r
        did[str(M)] = dict(estimate=delta - float(rl[M].mean() - base["change_aware"].mean()),
                           ci95=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
    for M, ph in zip(MS, holm(ps)):
        rows[str(M)]["p_holm"] = ph
        rows[str(M)]["outcome"] = change_outcome(rows[str(M)]["delta_vs_M5"], rows[str(M)]["ci95"], ph)
    out.write_text(json.dumps(dict(label="exploratory (M-scaling addendum: in-context)", commit=commit,
                                   rl_rerun_identical=True, score_M5=float(base["in_context"].mean()),
                                   by_M=rows, diff_in_diff_vs_rl=did), indent=2))
    with open(progress, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} saved {out} (commit {commit})\n")
    print(json.dumps({M: (round(r["delta_vs_M5"], 4), r["outcome"]) for M, r in rows.items()}),
          "diff-in-diff:", {M: round(v["estimate"], 4) for M, v in did.items()})


if __name__ == "__main__":
    main()
