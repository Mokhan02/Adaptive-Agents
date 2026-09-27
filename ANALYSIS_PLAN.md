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

_None._
