# Adaptive Agents

Detecting and adapting to regime change in competitive multi-agent settings.

When an adaptive agent's opponent changes strategy without warning, which
adaptation mechanism recovers fastest, and does the agent's internal
representation shift before, during, or after its behavior does?

The testbed is iterated Rock-Paper-Scissors against an opponent that switches
from Strategy A to Strategy B partway through, either abruptly or by linear
interpolation over a transition window.

The statistical analysis is pre-registered in [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md), and the agents and pretraining data in [AGENT_SPECS.md](AGENT_SPECS.md).

## Layout

```
src/regime/
  env.py        RPS payoffs, oracle, RegimeSwitchOpponent (hard / gradual switch, no-switch control)
  runner.py     run_episode -> EpisodeLog (actions, policies, expected + oracle reward, internal states)
  metrics.py    regret, recovery time, excess regret, representational drift
  pretrain.py   pretraining opponents for the in-context agent (held-out strategies excluded)
  agents/
    base.py       Agent interface: policy(), observe(), output_scores(), internal_state()
    baselines.py  reference agents for sanity checks (uniform, frequency counting)
scripts/
  sanity_check.py  multi-seed regret plot + recovery table for the reference agents
tests/
```

## Setup

```bash
pip install -e ".[dev]"          # add ,torch for the learned agents
pytest
python scripts/sanity_check.py --seeds 20 --transition 0
python scripts/sanity_check.py --seeds 20 --transition 100
```

Plots are written to `results/` (git-ignored).

## Conventions

- **Regret** is computed from expected rewards (the agent's policy against the
  opponent's true distribution) rather than realized rewards, so curves are not
  dominated by sampling noise.
- **Recovery time** is counted from the end of the transition, and is the
  number of rounds until the agent regains 90% of its pre-switch
  fraction-of-oracle performance for 20 consecutive rounds. It is undefined for
  agents with no pre-switch edge.
- **Excess regret** is oracle-normalized regret from the start of the switch
  through 100 rounds after it ends, net of the agent's pre-switch rate. Use it
  for gradual switches, where adaptive agents often recover during the
  transition and score 0 on recovery time.
- **Internal state and output scores** are logged after each round's update.
  Lead/lag compares their displacement over a per-agent horizon; see the analysis plan
  for why one-step drift is not used.

## Status

- [x] Pipeline + scripted opponent
- [x] No-switch control, continuous behavior signal, pre-registered analysis plan
- [ ] In-context agent, online fine-tuning agent, change-aware RL agent
- [ ] Lead/lag analysis (Tests A and B from the analysis plan)
- [ ] Stretch: second environment, severity sweep
