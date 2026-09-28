# Analysis plan (pre-registered)

Frozen before any of the three agents under comparison has been run. The commit
that introduces this file is the timestamp of record. Later changes go in
**Amendments** at the bottom with a date and reason; the sections above are not
edited.

## Hypotheses

**Primary (confirmatory):** in the RPS environment with a hard switch, the
in-context agent's internal state moves before its output distribution does.

Everything else is exploratory and reported without multiple-comparison
correction, labeled as such: the online fine-tuning and change-aware RL
agents, gradual switches, and the optional second environment.

The primary hypothesis is tested as a fixed sequence, which controls the
family-wise error rate at α = .05 without correction:

1. **Test A (the state responds to the switch).** If it fails, stop. The
   setup is under-powered (see Risks in the proposal), not a null on
   lead/lag.
2. **Test B (lead/lag is real).** Run only if Test A passes.

## Signals

All signals are logged after the agent observes round *t*.

- **State** `s_t`: the agent's internal state vector (hidden activations for
  the learned agents). Each dimension is standardized using its mean and
  standard deviation over rounds [T_w, switch_at) of the calibration control
  runs.
- **Behavior** `o_t`: the agent's continuous output scores (`output_scores()`,
  e.g. logits), centered to mean zero across actions. Sampled or argmax
  actions are never used as the behavior signal: a step function of a
  continuous belief would make any continuous state appear to lead by
  construction.
- **Displacement** at horizon h: `D_x(t) = ||x_t − x_{t−h}||₂`, computed the same
  way for state and behavior, with h set per agent (see Window parameters).

  This makes "drift" throughout the project a detector of *directional bias*,
  not of step magnitude. Under a stationary opponent the state takes a random
  walk whose steps cancel out. A switch makes the steps point consistently one
  way without making them bigger. One-step distance sees only magnitude, so it
  is blind to this by construction. On the reference frequency agent (window
  50, 200 seeds), the peak one-step drift after the switch beat the control's
  95th percentile in only 20% of runs, against 5% expected by chance.
  Displacement over h = 5 / 10 / 20 / 40 rounds caught it in 62 / 87 / 96 / 98%
  of runs.

## Runs

- **Seeds:** 100 switch runs and 100 no-switch control runs per condition.
  A two-sided sign test at α = .05 has ≥ 80% power when at least 65% of seeds
  lag in the same direction (it needs n = 90 for 65%, 49 for 70%, 30 for 75%
  and 20 for 80%). Runs are cheap, so power is not traded away.
- **Calibration seeds:** a separate 100 control runs and 100 switch runs
  (seeds 10000+) set the per-agent window parameters below. The analysis
  seeds (0–99) are never used for calibration.
- **Controls:** `RegimeSwitchOpponent.no_switch(STRATEGY_A, switch_at)`, which
  keeps the same `switch_at` anchor so every window lines up with the switch
  runs.

## Window parameters (per agent, from calibration seeds only)

- **Warm-up end T_w:** the first round after which the across-seed mean of
  windowed regret (20 rounds) in control runs stays within 10% of its mean over
  the final 100 rounds.
- **Displacement horizon h:** max(5, round(median recovery time / 2)) in
  calibration switch runs. For the reference agent this gives 38 / 2 ≈ 20,
  the horizon validated above. Scaling h to each agent's own adaptation time
  keeps a fast-adapting agent's signal from being smeared, and a
  slow-adapting agent's signal from being underpowered. The three agents have
  not been built, so a fixed h tuned on a proxy agent would be a guess.
- **Window half-width W:** twice the 90th-percentile recovery time in
  calibration switch runs. Recovery time is a behavioral metric only and does
  not involve the tested statistic.
- **Analysis window:** rounds [switch_at − W, switch_at + W]. If
  switch_at − W < T_w + h, switch_at is increased for that agent before any
  analysis run.
- **Why the in-context agent is primary:** in the fine-tuning agent, weight
  updates *are* the adaptation mechanism, so its internal state partly
  constitutes its behavior change. In the in-context agent, activations over
  the history window can move without any necessary change in the output.
  That makes it the cleanest case for asking whether representation
  anticipates behavior.
- **Maximum lag L:** ⌊W / 2⌋.

## Test A: the state responds to the switch

- Per-seed statistic: `max D_s(t)` over [switch_at, switch_at + W].
- Compare the 100 switch runs against the 100 control runs over the same
  window with a one-sided Mann–Whitney U test (switch > control) at α = .05.

## Test B: lead/lag

- Per seed, within the analysis window, z-score `D_s` and `D_o`, then compute
  their linear (non-circular) cross-correlation at lags −L…L. The lag estimate
  is the argmax; ties go to the smallest |lag|.
- **Sign convention:** a positive lag means the state moves first.
- **Primary test:** a two-sided exact sign test on the per-seed lags (zeros
  dropped) at α = .05.
- **Reported alongside, but not deciding:** the median lag with a bootstrap 95%
  CI (10,000 resamples) and a Wilcoxon signed-rank test.
- All seeds are included; none are excluded for "not responding." Seeds with
  no real response add random lags, which biases the result toward the null,
  never away from it.

Why not the proposal's permutation test: with one event per episode, shifting
one series against the other tests whether the series respond to the switch
at all, not the direction of the lag. And with circular cross-correlation, a
circular shift does not change the maximum over all lags, so the null would
equal the observed statistic.

## Outcome mapping

| Result | Reported as |
|---|---|
| A fails | State does not detectably respond: the setup is under-powered. Trigger the proposal's fallback and move to the richer environment. |
| A passes, B significant, median lag > 0 | Representation leads behavior by the median lag (with CI). |
| A passes, B significant, median lag < 0 | Representation lags behavior. |
| A passes, B not significant | No separable lead/lag. Report the CI as a bound, e.g. "any lead is under X rounds." |

## Secondary and exploratory analyses

- **Regret-derivative lag:** the same procedure with `D_o` replaced by the
  derivative of windowed regret, for continuity with the proposal.
- **Adaptation ranking:** recovery time (hard switch) and oracle-normalized
  excess regret (all switch severities) for each of the three agents.
- **Exploratory:** Tests A and B for the fine-tuning and RL agents, gradual
  switches (transition ∈ {25, 50, 100, 200}), and the second environment.

## Requirements on agents

Every agent implements `output_scores()`, returning continuous pre-decision
scores whose softmax (or other smooth map) gives `policy()`. Every learned
agent also returns a non-None `internal_state()`.

## Amendments

- **2026-09-27:** agent definitions, the in-context agent's pretraining
  distribution and held-out strategies, the choice of internal-state layer,
  and hyperparameter tuning rules are pre-registered in
  [AGENT_SPECS.md](AGENT_SPECS.md). This is an addition; nothing above is
  changed.
- **2026-09-27:** the in-context agent's state now has the latest-move
  component removed (see the AGENT_SPECS.md amendment of the same date).
  **Interpretive caveat for Test A, not a change to it:** for the in-context
  agent, Test A is a weak gate. Any function of the move window shifts when
  the opponent's move mix shifts: an untrained network detected
  pretraining-style switches 60% of the time (chance 5%). Passing Test A
  therefore shows the state responds to the switch, not that the model
  learned a representation of it. For this agent, the evidence lies in
  Test B. Test A is kept as pre-registered, because redesigning a test after
  learning it is uninformative would be a larger integrity problem than
  reporting a weak gate as weak.
- **2026-09-27, correction:** the "60%" in the Test A caveat above came from
  a preliminary in-sample estimate. With the pre-registered move offsets, an
  untrained network detects 39% of pretraining-style switches (chance 5%).
  The caveat stands: any function of the move window responds to the switch
  to some degree. Details are in AGENT_SPECS.md.
- **2026-09-27, calibration and analysis rules, fixed before any calibration
  run.** These fill gaps in the sections above; nothing above is reversed.
  - **Seeds.** Calibration: switch runs 10000–10099, controls 10100–10199.
    Analysis: switch runs 0–99, controls 1000–1099. Controls use seeds
    disjoint from switch runs, so the two groups in Test A are independent.
  - **Recovery time** is `metrics.recovery_time` with its defaults (20-round
    window, 90% threshold, sustained 20 rounds, 50-round baseline).
  - **Censoring.** A run that does not recover by the end of the episode,
    or has no pre-switch edge (so recovery is undefined), is *censored*. It
    counts as longer than every observed recovery time when taking the
    median and 90th percentile. If the 90th percentile is censored (more
    than 10% of calibration runs censored), W is undefined, and that agent's
    Tests A and B are **not runnable**. This is reported as such, not as a
    null. The agent still appears in the adaptation ranking, with its
    censored count reported.
  - **T_w (clarified).** "Within 10% of its final level" is ill-posed when
    the final level is near zero. T_w is the first round after which the
    across-seed mean of 20-round windowed regret in calibration controls,
    m(t), stays within 0.1 · |m(0) − m_final| of m_final, where m_final is
    the mean over the last 100 rounds.
  - **Window floor.** W = max(2 · p90 recovery, 3h). An agent that recovers
    almost instantly would otherwise get W ≈ 0. A displacement at horizon h
    responds over about h rounds, so the window must span a few h. L = ⌊W/2⌋
    as before.
  - **Episode length.** switch_at = max(200, T_w + h + W) and
    n_rounds = max(600, switch_at + W + 1), so the whole analysis window
    lies inside the episode and after warm-up. Both are recorded per agent
    in the calibration output.
  - **Standardization.** Per-dimension mean and standard deviation of the
    state, pooled over calibration control runs and rounds [T_w, switch_at).
    Dimensions with zero variance are left unscaled. Behavior (output
    scores) is centered across actions only.
  - **Lag ties.** The argmax lag with the smallest |k| wins. A tie between
    +k and −k is recorded as 0, meaning no direction.
  - **Lag convention, precisely:** the lag is the k maximizing
    corr(D_s(t), D_o(t + k)). If the state's displacement rises k rounds
    before the behavior's, the peak is at +k. This convention is tested on
    planted signals (`tests/test_analysis.py`).
  - **One confirmatory run.** Calibration outputs and the analysis code are
    committed and tagged before seeds 0–99 run. The results file records the
    commit hash. Any later fix goes in an amendment with its reason.
  - **Calibration never computes Tests A or B.** It only derives T_w, h, W,
    L, the standardization statistics and the episode length.
