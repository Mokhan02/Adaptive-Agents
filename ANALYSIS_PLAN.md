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
- **2026-09-27, calibration bug fix, before any analysis run.**
  - **The bug.** Standardization statistics were computed over control-run
    rounds [T_w, 200). This range is empty when T_w ≥ 200, which happened
    for the RL (T_w = 249) and fine-tuning (T_w = 330) agents, and it
    produced NaN statistics.
  - **The fix.** Controls never switch, so the post-warm-up baseline is
    [T_w, end of episode). This matches the intent of "[T_w, switch_at)",
    and the fix applies to all three agents. Calibration is deterministic,
    so it was rerun with the fix: recovery times, T_w, h, W, L, switch_at
    and n_rounds are unchanged by construction, and only the
    standardization statistics change. Each calibration file also stores
    the control mean-regret curve behind T_w.
  - **Limitation noted at the same time (exploratory agents only).** For
    the RL and fine-tuning agents, T_w exceeds the calibration switch round
    (200). Their recovery times, and therefore h and W, were measured before
    they had finished warming up, while their analysis runs switch after
    warm-up (at rounds 531 and 941). The rule is applied as pre-registered.
    Their results carry this caveat. The primary agent (T_w = 31) is
    unaffected.
- **2026-09-27, before the confirmatory run.**
  - **RL T_w is a band-edge artifact.** The RL agent's control regret is
    flat from about round 100 (0.181 → 0.179 through round 200), but it
    settles high (0.167), so the 10% band is only ±0.023. A noisy excursion
    to 0.190 near round 250 sets T_w = 249 and pushes its switch to round
    531. The rule is applied as written, and this caveat is recorded.
  - **Adaptation ranking setup.** Tests A and B use each agent's own
    switch_at, which makes recovery times incomparable across agents. The
    ranking therefore uses the test opponent as specified in AGENT_SPECS.md
    (hard switch at round 200 of 600) on analysis seeds 0–99, the same for
    all agents. Recovery time is reported as a censored median with the
    censored count; excess regret as a mean ± SE.
  - **Secondary regret lag.** The behavior signal is replaced by
    |Δ m(t)|, the absolute first difference of the 20-round windowed
    regret. Everything else follows Test B.
  - **Also reported as exploratory for the in-context agent:** Test B on the
    raw (uncorrected) state.
  - **The confirmatory script refuses to run** unless the working tree is
    clean, HEAD carries the tag `confirmatory-v1`, and no results file
    exists. The results file records the commit hash.
- **2026-09-27, additions before tagging `confirmatory-v1`.**
  - **Ranking: two variants on the same seeds (0–99).** The headline is the
    common structure (switch at round 200 of 600): it is the test opponent
    as specified, and every agent gets the same 400-round recovery horizon.
    The sensitivity variant uses each agent's own calibrated switch_at, so
    every agent is past warm-up. It reuses the Test A/B switch runs, but the
    recovery horizons differ by agent (400, 261 and 529 rounds). The
    fine-tuning agent is still warming up at round 200 in the headline, so
    its headline recovery mixes learning with adapting. If the two
    variants order the agents differently, that is reported as a finding.
  - **Recovery outcomes are counted separately:** recovered, not recovered
    by episode end, and no pre-switch edge (recovery undefined). All three
    counts are reported for every agent and variant.
  - **Secondary regret lag.** |Δ m(t)| has two bumps for a hard switch (the
    jump, then the recovery), and the argmax may land on either. The mean
    cross-correlation curve is reported alongside the lags. This analysis
    is not evidence for or against the primary result.
  - **Edge case in the outcome mapping.** If the sign test is significant
    but the median lag is 0, the result is reported as "direction per the
    nonzero majority, magnitude below resolution".
  - **Dry run.** `--dry-run` runs the whole pipeline on an allowed non-A/B
    pair, (0.1, 0.8, 0.1) → (0.1, 0.1, 0.8), with 6 seeds, writing to a
    scratch path. It was run once, before tagging, to confirm the pipeline
    completes and writes every output. Its numbers are not results.
  - **The results file is committed unedited**, including any failed test.

## Post-results notes (2026-09-27)

Written after the confirmatory run (`results/confirmatory/results.json`,
tag `confirmatory-v1`, commit `5dcdc62`) and before any follow-up analysis.
The results file is unchanged; these notes govern how it is reported.

**The primary result as pre-registered.** Test A passed (U = 8527 of 10,000,
AUC 0.85). Test B: 92 of 100 seeds negative, median lag −2.5, 95% CI [−3, −2].
The outcome is "representation lags behavior". The claim is narrow: one
agent, one game, a hard switch, and the state after latest-move correction.
Test A is a weak gate for this agent, so the evidence is in Test B.

**Caveats reported with it:**
1. **Sign flip without the correction.** The exploratory raw-state lag is
   +4.5 (CI [2, 7]). The pre-registered corrected definition decides the
   headline. The amendment's "not conditional on results" clause, written
   before any model saw A or B, is what makes that defensible.
2. **Possible saturation artifact.** The output scores are bounded and
   saturate once the prediction flips; the state is not bounded. A
   displacement peak can come earlier on a saturating signal even with
   simultaneous onsets. This is tested below.
3. **The secondary regret lag is flat** (49 / 50 / 1) and is not evidence.

**Reclassification of the exploratory Test B results.** For both the RL and
fine-tuning agents, all 100 lags are exactly 0. The sign test has no
nonzero observations, so Test B is **not testable by construction**, not
"no separable lead/lag". The results file's `outcome` string for these
agents applied the mapping mechanically and is superseded here.
- RL: expected and pre-recorded (its state contains its outputs).
- Fine-tuning: **a design error found after the run.** The state (the
  hidden layer) is one linear map from the logits, which is the same flaw
  the spec avoided for the in-context agent's final layer.
- **Deviation from the spec:** the fine-tuning agent's weight-displacement
  signal (AGENT_SPECS.md, section 2) was not computed by the confirmatory
  script.

**Reading the ranking.** Excess regret subtracts each agent's own
pre-switch regret rate. An agent that plays loosely throughout (RL: steady
regret 0.167 per round, versus in-context 0.015) pays little relative to
its own baseline. The pre-registered ranking is reported with this sentence
beside it, and total regret is added as an exploratory column.

**The hold-out claim is not yet established.** The in-context agent's
excess regret was 11.88 on tuning episodes and 20.7 on A → B, the same
metric. But the tuning episodes differ in two ways: half were gradual
switches, and pairs needed only TV ≥ 0.2, while A → B differs by 0.4. Until
the control below runs, the write-up says "consistent with", not "shows".

### Exploratory follow-ups (defined before running; all labeled exploratory)

Run in this order. Follow-up 1 uses no real seeds.

1. **Planted saturation check.** A latent belief moves linearly from p to q
   over 50 rounds, matching the context window, starting at the switch. The
   state is a noisy linear projection of the belief. The behavior is
   PAYOFF · softmax(β · log belief) + noise, for β ∈ {1, 2, 4, 8}. At β = 1
   the output is linear in the belief; larger β saturates it. The true lag
   is zero. The pipeline runs with the in-context calibration (h = 14,
   W = 86, L = 43). If the estimated lag goes negative as β grows, the
   pipeline is biased by saturation.
2. **Unsaturated behavior signal.** Test B on analysis seeds 0–99, with
   behavior = the final position's logits (the model's raw output, before
   softmax), centered across actions. The runs are deterministic
   reproductions of the confirmatory episodes.
3. **Untrained-network control.** An untrained network (`torch.manual_seed(0)`,
   the same architecture), with move offsets estimated as for the frozen
   model and standardization from calibration controls (seeds
   10100–10199). Test B on seeds 0–99 at the in-context h, W and L, for the
   corrected and the raw state.
4. **Sensitivity to h.** Test B for the in-context agent at h = 7 and
   h = 28 (W and L unchanged).
5. **Onset comparison.** For each signal, the threshold is the 95th
   percentile of its displacement in control runs (seeds 1000–1099) over
   the analysis window. The onset is the first round in [switch_at,
   switch_at + W] at which the displacement exceeds that threshold. Per
   seed, the difference is onset(behavior) − onset(state), so positive
   means the state moves first, as in Test B. Seeds where either onset is
   missing are counted and excluded. Sign test on the nonzero differences.
6. **Total regret.** The sum of per-round regret over 600 rounds, for the
   headline ranking structure (switch at 200, seeds 0–99), all three agents.
7. **Fine-tuning weight displacement.** The state is the flattened weight
   vector, unstandardized. Test B at the fine-tuning calibration's h, W, L
   on seeds 0–99.
8. **Hold-out control.** 100 hard-switch episodes at round 200 of 600
   between allowed pairs with TV ∈ [0.35, 0.45], with p and q drawn by the
   pretraining rules. Seeds 7,000,000–7,000,099. Excess regret for all three
   agents, to compare with A → B.

### Exploratory follow-up results (2026-09-27)

Outputs are in `results/exploratory/`. All are exploratory. Reruns of
confirmatory episodes reproduced the confirmatory lags exactly.

| # | Check | Median lag [95% CI] | +/−/0 | Sign p |
|---|---|---|---|---|
| — | Confirmatory (reference) | −2.5 [−3, −2] | 8/92/0 | 3e−19 |
| 1 | Planted, true lag 0, saturating output β = 1, 2, 4, 8 | 0 at every β | e.g. β = 8: 43/36/21 | ≥ 0.2 |
| 2 | Behavior = logits (no saturation) | −2.0 [−3, −2] | 6/94/0 | 2e−21 |
| 3 | Untrained network, corrected / raw state | 0 / 0 | 0/0/100 both | 1 |
| 4 | h = 7 / h = 28 | −2.5 / −2.0 | 24/76/0, 1/99/0 | 2e−7, 2e−28 |
| 5 | Onset difference (2 seeds missing an onset) | −3.0 [−5, −1] | 36/62/0 | 0.011 |
| 7 | Fine-tuning, weight-vector state | −5.0 [−6, −4] | 6/85/9 | 6e−19 |

- **Saturation (1, 2):** it does not produce the negative lag. The planted
  zero-lag case stays at 0 under any saturation, and the unsaturated logit
  behavior gives the same answer (−2.0, 94 of 100 seeds negative).
- **Learning (3):** the untrained network's state and outputs move in
  lockstep (lag exactly 0), so the −2.5 (and the raw state's +4.5) arise
  with training, not from the architecture or the pipeline. This control
  is weak, because a lockstep network is not a matched comparison.
- **Robustness (4, 5):** the sign holds from h = 7 to 28, and under a
  threshold-based onset definition that doesn't use cross-correlation.
- **Fine-tuning weights (7):** the weight vector lags the logits by about
  5 rounds, the same direction as the in-context result. Mechanistically,
  the input window changes the logits immediately, and gradient steps catch
  up afterwards. h = 83 for this agent, so the lag is small relative to
  the resolution.

**Regret (6, 8), 100 seeds each, 200/600 hard switch:**

| Agent | Total regret A → B | Excess A → B | Excess, control pairs (TV 0.35–0.45) |
|---|---|---|---|
| In-context | **24.5 ± 0.9** | 20.7 ± 2.2 | 22.8 ± 4.3 |
| Change-aware RL | 112.5 ± 1.4 | **7.6 ± 2.4** | 21.5 ± 3.8 |
| Fine-tuning | 139.7 ± 2.3 | 82.7 ± 3.6 | 57.0 ± 5.4 |

- **Total regret reverses the top of the pre-registered ranking.** The
  in-context agent loses 4–5× less in total than RL. RL's excess-regret
  win comes from subtracting its own loose baseline.
- **The hold-out claim is refuted, not supported.** On allowed pairs of
  the same size and switch type, the in-context agent's excess regret is
  22.8 ± 4.3, indistinguishable from A → B (20.7). The rise from the tuning
  score (11.88) is explained by the switch type and size, not by the
  excluded strategies. The write-up must not claim the hold-out affected
  performance.
- **A → B is unusually easy for RL** (7.6 against 21.5 on comparable
  pairs). On comparable pairs, in-context and RL are tied on excess regret.

## Structural-null check (2026-09-28, defined before running)

**Why.** Study 2's dry run (`results/early_detection/dry_run/`: the non-test
pair (0.1, 0.8, 0.1) → (0.1, 0.1, 0.8), seeds 39000–39059) gave a
normalized-timing difference of −5.1 rounds at 50% between layer 2 and the
output, and −6.0 against the logits. Layer 2 determines the logits in the
same round, so no information lag is possible there. The method is measuring
the *shape* of displacement curves: a 64-dimensional residual's displacement
is dominated by directions the output never reads. Study 1 compared the same
kinds of signal with cross-correlation and had no layer-2 control.

**The early bump is already compromised.** In the same dry run, layer 2
showed the "state first at 10%" pattern (+1.2 rounds against the output)
that the bump suggested for layer 1. A real early signal is structurally
impossible at layer 2. So the bump is recorded as a curve-shape effect,
independently of the check below.

**The check.** Study 1's exact pipeline, with only the state swapped:
- **State:** the layer-2 residual at the final position, latest-move
  corrected using the frozen model's own layer-2 offsets (fitted by the same
  pre-registered procedure as layer 1's, on pretraining-distribution
  episodes), and standardized per dimension on study 1's calibration
  controls (10100–10199, rounds [T_w, end)), as layer 1 was.
- **Behavior:** the centered output scores.
- **Lag:** h = 14, W = 86, L = 43, argmax of the linear cross-correlation,
  the same tie rules, and Test B's summary.
- **Seeds 0–99.** A reproduction check first asserts that the rerun gives
  study 1's layer-1 lags exactly.
- **Also reported:** raw layer 2 (uncorrected, unstandardized, as study 1's
  raw-state exploratory analysis).

**Why seeds 0–99 are the right seeds.** This is a structural diagnostic of
a frozen model and an already-run pipeline, not a new confirmatory test. It
asks what this exact pipeline would have said about a signal whose true
information lag is known to be zero, on the same data. If it later grows
into a claim of its own (a redesigned study 2), that claim needs fresh
seeds.

**Interpretation, fixed now.** Study 1: median −2.5, 95% CI [−3, −2],
92 of 100 seeds negative.
- **Comparable, so the artifact explanation is supported:** corrected
  layer-2 median lag in [−3.5, −1.5], its CI overlapping [−3, −2], and at
  least 80% of nonzero seeds negative. The headline is then reframed:
  "residual-stream displacement peaks after output displacement, a pattern
  also present at a layer with no possible information lag, so it does not
  support a claim about representational timing."
- **Meaningfully different, so the artifact explanation is weakened:**
  median lag above −1.0, *or* fewer than 60% of nonzero seeds negative, with
  a CI that does not overlap [−3, −2]. This does not clear layer 1: the
  difference between the layers then needs its own explanation.
- **Anything else is intermediate:** "partly explained by curve shape".
  Both numbers are reported side by side.

**Study 2 is on hold.** Its draft pre-registration (PREREG_EARLY_DETECTION.md)
is not frozen. Its negative-control expectation of "about 0" was wrong.
Any redesign would compare the state against a structural null (layer 2,
or a dimension-matched projection of the state) rather than against zero,
and it will be specified only after this check is in.

### Structural-null check: result (2026-09-28)

`results/exploratory/structural_null_layer2.json`. The rerun reproduced
study 1's layer-1 lags exactly.

| Signal vs output | Median lag [95% CI] | +/−/0 |
|---|---|---|
| Layer 1, corrected (reference) | −2.5 [−3, −2] | 8/92/0 |
| Layer 2, corrected | −2.0 [−2, −1] | 11/89/0 |
| Layer 2, raw | +2.0 [0, 5] | 57/25/18 |

**Verdict under the rule above: comparable, so the artifact explanation is
supported.** The median is in [−3.5, −1.5], 89% of nonzero seeds are
negative, and the CIs overlap. The overlap is marginal: the CIs touch at −2,
and layer 2's median is 0.5 rounds smaller. Raw layer 2 reproduces the sign
flip.

**The headline is reframed**, as pre-registered: "residual-stream
displacement peaks after output displacement, a pattern also present at a
layer with no possible information lag, so it does not support a claim
about representational timing." The pre-registered outcome and the results
file are unchanged. This note governs how they are reported. The robustness
checks establish a property of the displacement curves, not of
representational timing. The proposal's early-warning question remains open.

**Clarification of the verdict (2026-09-28): the endpoint convention.** The
rule's "CI overlapping [−3, −2]" did not specify whether intervals that
touch at an endpoint overlap. The analysis code treats intervals as closed
(`lo <= -2 and hi >= -3`), so touching counts. That code was written before
the run but committed together with the result, so the record cannot show
the ordering. Under an open-interval reading, the result is
**intermediate** ("partly explained by curve shape"), because the
"meaningfully different" condition also fails (median −2.0 is not above
−1.0, and 89% of seeds are negative). The verdict is therefore reported as
on the boundary between "comparable" and "intermediate". Under either
reading, the lag does not support a representational-timing claim. Future
rules on integer-valued lags must state their endpoint convention.

## Standing conventions (2026-09-28; apply to every rule written from now on)

1. **State the endpoint convention for every comparison.** Say whether
   intervals are closed or open, and whether inequalities are strict. Most
   of this project's statistics are whole numbers (lags, delays) or bootstrap
   percentiles of whole numbers, so ties at a boundary are common, not rare.
   The structural-null verdict landed exactly on such a boundary and could
   only be reported as "on the boundary". Pre-registering interpretation rules
   doesn't remove every judgment call, but it makes the remaining ones
   visible. Making them visible *in advance* is this convention's job.
2. **Commit rule-implementing code before the run it governs,** separately
   from the results, so the record shows the order.
3. **Give relative tolerance bands an absolute floor.** A band defined as
   a fraction of a level that can be near zero (for example, 10% of a
   final regret or reward) is ill-posed there. Study 1's T_w rule hit this
   for the RL agent (see the band-edge note). New rules use
   max(relative band, absolute floor).

## Dimension diagnostics (2026-09-28, defined before running)

**Question.** Is the layer-1 vs output lag an artifact of the dimension
mismatch, with displacement of a 64-dimensional state dominated by
directions the output never reads? Both diagnostics use study 1's pipeline
(corrected, standardized layer-1 state; centered output scores; h = 14,
W = 86, L = 43; seeds 0–99; lag = argmax of linear cross-correlation, the
same ties). Only the state is replaced by a projection of it. Both are
exploratory diagnostics on existing seeds.

The centered logits have 2 degrees of freedom (3 actions, centered), so the
readout subspace is 2-dimensional, and the sweep starts at k = 2.

### D1: readout projection

- **Fit:** ridge regression (λ = 1e-3 × trace / d) of the centered logits on
  the standardized corrected layer-1 state. The data are the frozen agent's
  final-position states on 100 pretraining-distribution episodes (seeds
  4,100,000–4,100,099), rounds ≥ 64 (full context, after warm-up). The
  readout subspace Q is an orthonormal basis of the coefficient matrix's
  column space (rank 2).
- **Validity condition (checked first, closed ≥):** the held-out R² of the
  fitted regression predicting centered logits is ≥ 0.70 on both 50 held-out
  pretraining episodes (4,200,000–4,200,049) and study 1's calibration
  switch runs on A → B (10000–10099, rounds ≥ 64). If either is below 0.70,
  D1 is reported as **invalid** (the subspace doesn't carry the output's
  information on this data), with no lag interpretation.
- **Statistic:** Test B's summary for the projected state (s · Q) vs the
  output, on seeds 0–99.
- **Interpretation** (study 1's layer 1: −2.5 [−3, −2]):
  - **Shrinks (mismatch explanation supported):** median > −1.0 (strict)
    *and* the CI's upper end is ≥ 0 (closed).
  - **Reverses:** median > 0 (strict) *and* the CI's lower end is > 0
    (strict), meaning the readout moves before the output. This would be
    interesting, and it would be exploratory.
  - **Persists:** median ≤ −2.0 (closed) *and* the CI's upper end is < 0
    (strict).
  - **Anything else:** partial.

### D2: random-projection dimension sweep

- **Projections:** for k ∈ {2, 4, 8, 16, 32, 64}, 20 random k-dimensional
  orthonormal projections of the standardized corrected layer-1 state
  (`numpy.random.default_rng(6_100_000)`, QR of Gaussian matrices, drawn in
  that order). At k = 64 a projection is a rotation, which leaves
  displacement norms unchanged, so all 20 must reproduce study 1's lags
  exactly. This is asserted as a sanity check.
- **Statistic per projection:** the median lag over seeds 0–99. Per k: the
  mean and the range of |median lag| across the 20 projections.
- **Interpretation:** a one-sided Spearman correlation between k and |median
  lag| over the 120 (k, projection) pairs, testing whether larger k gives a
  larger |lag|.
  - **Shrinks toward 0 (mismatch supported):** Spearman p < 0.05 (strict)
    *and* the mean |median lag| at k = 2 is ≤ 1.0 (closed), which is 40% of
    layer 1's 2.5.
  - **Partial dose-response:** p < 0.05, but the k = 2 mean is > 1.0.
  - **Flat (not dimension per se):** p ≥ 0.05. This points to specific
    directions, not dimensionality.
- **Also reported:** D1's readout lag against the k = 2 random distribution,
  to show whether the readout directions behave differently from random 2-D
  projections.

### Dimension diagnostics: results (2026-09-28)

`results/exploratory/dimension_diagnostics.json`. The code was committed
(`46f7a69`) before the run. The rerun reproduced study 1's layer-1 lags
exactly, and all 20 k = 64 rotations did too.

**D1, readout projection: invalid.** The held-out R² was 0.756 on
pretraining episodes but **−0.018 on the A → B calibration runs**, below
the 0.70 condition. A linear readout of layer 1 fitted on the pretraining
distribution does not carry over to the held-out test strategies. Two
likely reasons: A and B lie in the excluded region of strategy space, and
within A → B episodes the logits vary little, which makes R² harsh. Its lag
(−3.0 [−4, −2]) is not interpreted, per the rule.

**D2, dimension sweep: flat, not dimension per se.**

| k | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Mean \|median lag\| over 20 projections | 3.27 | 2.40 | 2.45 | 2.35 | 2.33 | 2.50 |
| Range of medians | −10.5 to 10.5 | −3 to −1 | −3 to −1.5 | −3 to −2 | −3 to −2 | −2.5 |

One-sided Spearman ρ = −0.037, p = 0.657. Random slices as small as 4
dimensions lag like the full state; at k = 2, estimates become noisy rather
than smaller. **This refutes the dimension-mismatch explanation** given in
the structural-null section and the README, which has been corrected. The
lag is a property of almost every direction of the residual. The output's
readout directions, which by construction don't lag the logits, must be
atypical.

**Open hypothesis, not tested.** The model applies LayerNorm before the
unembedding. If the residual's overall magnitude drifts slowly after a
switch, every raw projection inherits that drift, while the output (which
reads normalized residuals) does not. A test would compute displacement on
the per-round normalized residual. It would be defined here before it runs.

## LayerNorm diagnostic (2026-09-28, defined before running; the last mechanism check)

**Hypothesis.** The residual's overall magnitude drifts slowly after a
switch. Every raw direction inherits that drift, but the output reads a
LayerNorm-normalized residual and does not.

**The check.** Study 1's pipeline, with one step inserted first. Each
round's *raw* residual (final position) is normalized the way the model's
LayerNorm does it, without the affine part: center across the 64
dimensions, divide by their standard deviation (ε = 1e-5). Then, as in
study 1:
- latest-move correction, with per-move means estimated *in the normalized
  space* by the pre-registered procedure (200 pretraining-distribution
  episodes, seeds 4,000,000+, full windows);
- per-dimension standardization on calibration controls 10100–10199, rounds
  [T_w, end);
- displacement at h = 14, lag against the centered output scores (W = 86,
  L = 43), Test B's summary, seeds 0–99.

A reproduction check first asserts that the unnormalized layer-1 lags equal
study 1's. **Layer 1 is primary.** Layer 2 is reported alongside and does
not change the verdict.

**Interpretation** (study 1's layer 1: −2.5 [−3, −2]; endpoints as
stated):
- **Supports magnitude drift through LayerNorm:** median in [−1.0, 1.0]
  (closed) *and* the 95% CI contains 0 (closed: lo ≤ 0 ≤ hi). The write-up
  then calls it one plausible mechanism consistent with the data, not the
  only one.
- **Disconfirmed:** median ≤ −1.5 (closed) *and* the CI's upper end is < 0
  (strict). The write-up then says the mechanism is unresolved.
- **Anything else:** ambiguous. The write-up says the mechanism is
  unresolved.

**Stop rule.** Whatever the outcome, no further mechanism diagnostics are
run. The next step is the write-up.

### LayerNorm diagnostic: result (2026-09-28)

`results/exploratory/layernorm_diagnostic.json`. The code was committed
(`a77fdea`) before the run, and the reproduction check passed.

| State (LayerNorm-normalized) vs output | Median lag [95% CI] | +/−/0 |
|---|---|---|
| Layer 1 (primary) | −3.0 [−3, −2] | 6/94/0 |
| Layer 2 | −2.0 [−2, −1] | 7/93/0 |

**Verdict: disconfirmed** (median ≤ −1.5, CI upper end < 0). Removing the
residual's per-round magnitude does not remove the lag, so magnitude drift
through LayerNorm is not the mechanism. **The mechanism is unresolved.**
Per the stop rule, no further mechanism diagnostics are run.

---

# Study 3: a reactive opponent (draft, 2026-09-28; frozen at tag `study3-v1`)

_Draft revision 2026-09-28, before any code or run: the power rule now carries a variance inflation factor and a justification for its SESOI, and the M-selection window's start is explained._

Studies 1–2 used a scripted opponent: nonstationary, but not competitive,
since it never reacted to the agent. Study 3 replaces it with an opponent
that best-responds to the agent's own recent play (fictitious play). This
is the structural property the proposal's "competitive multi-agent" framing
needs. The standing conventions above apply: every comparison states its
endpoints, and rule-implementing code is committed before the run it
governs.

## Question and the agents' handicap

**Do agents built for scripted nonstationarity transfer to an opponent that
reacts to them?** The three agents are used **frozen**, exactly as in
`results/tuning/winners.json` and `models/icl_frozen.pt`, with no new
pretraining or tuning. This is a transfer test, and each agent carries a
handicap stated here as part of the hypothesis, not as a caveat:
- **The in-context agent** sees only the opponent's moves (AGENT_SPECS.md:
  "the opponent does not react to the agent"). Against this opponent, the
  agent's *own* history is what predicts the opponent, and the agent cannot
  see it. Its pretraining never included a reactive opponent.
- **The fine-tuning and RL agents** were tuned on scripted switches.
- A poor result for any agent is evidence about **transfer**, not a verdict
  on its adaptation mechanism in general.

## Environment

**The fictitious-play opponent** (`env.py`, `FictitiousPlayOpponent`):
- **Before round M** (fewer than M agent moves seen): it plays uniformly at
  random.
- **From round M on:** it takes the empirical distribution p of the agent's
  last M *sampled* moves. Its best response is argmax(PAYOFF · p), with
  ties broken uniformly among the tied actions. Its mixed strategy each
  round is (1 − ε) · (the best response, or uniform over tied best
  responses) + ε · uniform.
- **Observation:** no lag on either side. The opponent sees the agent's
  move from round t before round t + 1, and the agent sees the opponent's
  move likewise. Lag and noise are a later, separate axis. When added, the
  lag is symmetric by default, and any asymmetry is a stated design choice.
- **ε = 0.1.** **M** is chosen by the rule below.
- **Episodes are 1,000 rounds with no switch.** That is the primary design,
  (a).

**Exploratory bridge condition, design (b), run only after the primary
analysis:** the scripted opponent plays A for 200 rounds, then switches to
the fictitious-play opponent. It is reported as exploratory and never
pooled with the primary. It mixes "adapting to a change" with "adapting to
a reactive opponent", which the primary design keeps apart.

## Metric

- **Expected reward per round:** the agent's policy against the opponent's
  mixed strategy that round, which is well defined even though that
  strategy depends on history. This keeps the variance reduction used
  throughout the project.
- **Nash value (0)** is the reference point: uniform play earns exactly 0
  in expectation against any opponent.
- **Oracle regret is dropped for this study.** Best-responding myopically
  to a reactive opponent is not optimal play, so regret against it is not
  meaningful.
- **Per-seed score:** mean expected reward over rounds [T_w, 1000).
- Nothing is applied retroactively to studies 1–2.

## Calibration (seeds 50000–50199; reference agents only)

Calibration never runs the three frozen agents, so it cannot preview the
answer. The reference agents are: uniform (Nash), always-rock, and the
frequency counters with windows 20 and 50 and with full history (study 0's
baselines).

1. **Choose M.** For each M ∈ {5, 10, 20, 50}, in that order, run every
   reference agent on 100 seeds (50000–50099). Score each run by its mean
   expected reward over the fixed window [300, 1000). The window starts at
   300 to stay conservatively past any plausible warm-up for the reference
   agents, whose slowest settling in study 0 was about 110 rounds (the
   full-history counter). Fixing it independently of T_w avoids circularity,
   since T_w depends on M. M is **the smallest M
   at which every pair of reference agents is separated by more than 2 SE**
   (strict >), where SE is the standard error of the difference in means.
   If no M qualifies, use M = 20, and record that the rule failed.
2. **Warm-up T_w** at the chosen M, on seeds 50100–50199, for each
   reference agent. Take m(t), the across-seed mean of the 50-round trailing
   mean expected reward, and m_final, its mean over rounds [800, 1000).
   T_w is the first round after which
   |m(t) − m_final| ≤ max(0.1 · |m(0) − m_final|, 0.01)
   (closed ≤) holds for every later round. The absolute floor of 0.01 keeps
   the band well-defined when m_final is near 0. **The study's T_w is the
   maximum over reference agents.** If T_w > 700, the episode length is
   extended so that at least 300 rounds follow it.
3. **Power and seed count.** From the calibration runs at the chosen M,
   take σ_ref, the **largest** per-seed SD of the score across the reference
   agents. Multiply it by k = max(1, 0.99) = 1. Here k is a variance
   inflation factor from study 1's public data: the most variable frozen
   agent's per-seed SD of total regret (fine-tuning, 22.9) divided by the
   most variable reference agent's under the same scripted runs (full-history
   counter, 23.2). This guards against the frozen agents being more variable
   than the references. In study 1 they weren't, but that estimate comes
   from the scripted environment. N per agent is the smallest of
   {100, 200, 400, 800} giving at least 80% power (normal approximation,
   two-sided, two-sample) at α = 0.05/3 (Holm's strictest step) for a
   difference of **0.02 reward per round**. If none qualifies, N = 800,
   declared underpowered. The run also reports the *achieved* power from the
   observed SDs; N is not changed after the fact.
   - **Why 0.02 (SESOI).** In study 1 the frozen agents' pairwise
     total-regret gaps were 0.147 (in-context vs RL) and 0.045 (RL vs
     fine-tuning) per round. So 0.02 is under half the smallest gap they
     showed: a difference smaller than that would be negligible next to how
     these agents differed before.

## Hypotheses and tests (test seeds 60000 to 60000 + N − 1, each agent)

**H1, directional (a mechanistic prediction on n = 3, not a statistical
test).** The more predictable an agent's play, the more the opponent
exploits it. Predictability is measured by **entropy**: the entropy of the
agent's own empirical move distribution over the trailing M rounds,
averaged over rounds [T_w, 1000) and seeds, the same window the opponent
conditions on.
- **Prediction:** the ordering of the three agents by entropy equals their
  ordering by score (Spearman over 3 agents = +1). Specifically, **the
  in-context agent has the lowest entropy and the lowest score.** Its tuned
  temperature (0.035) makes it the sharpest player. This reverses its
  first place on total regret in study 1.
- **Status:** confirmed only if both orderings match exactly. With three
  agents this is an ordinal check and is reported as such, not with a
  p-value.

**H2, pairwise score differences (statistical).**
- For each agent pair: the difference in mean score, with a bootstrap 95%
  CI (10,000 resamples of seeds within each agent) and a two-sided
  bootstrap p-value. Holm correction across the 3 pairs.
- Outcomes per pair, with SESOI = 0.02:
  - Holm p < 0.05 and |difference| ≥ 0.02: **"A above B by at least the
    effect of interest"**.
  - Holm p < 0.05 and |difference| < 0.02: **"a real difference below the
    effect of interest"**.
  - Holm p ≥ 0.05 and the CI strictly inside (−0.02, 0.02) (open):
    **"equivalent within the margin"**.
  - Anything else: **"inconclusive"**.

**H3, each agent against Nash.** A bootstrap 95% CI of each agent's mean
score, with Holm across the 3 agents:
- Upper end < 0 (strict): **"exploited"**.
- Lower end > 0 (strict): **"exploits the opponent"**.
- Otherwise: **"not distinguishable from the Nash value"**.

**Did the ranking change?** Study 3's order, taken over the pairs H2
separates, is compared with **both** of study 1's orderings: the
pre-registered excess-regret order (RL > in-context > fine-tuning) and the
exploratory total-regret order (in-context > RL > fine-tuning). The ranking
"changed relative to" an ordering if at least one pair H2 separates points
the other way. Both comparisons are reported.

## Play-trajectory diagnostics (reported, not tested)

Fictitious play in zero-sum games converges in empirical frequency, not in
play, so cycling is expected. A null on the score must not be read as
"nothing happened". Reported per agent:
- the trailing-M entropy of the agent's moves and of the opponent's;
- the dominant cycle period: the lag (2–200) of the first local maximum
  above 0.2 in the autocorrelation of the opponent's best-response sequence
  (one-hot, averaged over actions), or "none";
- example trajectories for 3 seeds.

## No internal-state claims

Study 3 makes none. There is no switch to anchor a lead/lag analysis, and
the last internal-state lag needed three structural nulls to interpret
(and turned out to be a measurement property). Any future internal-state
analysis in this environment would start from a structural null.

## Process

1. This section, the opponent code and its tests are committed: always-rock
   earns −(1 − ε) = −0.9 per round in expectation once M moves are seen;
   uniform play earns exactly 0; ties and the pre-M rounds are uniform.
2. Calibration (reference agents only). Its outputs (M, T_w, N, with the
   power table) are committed.
3. **Dry run** on a deliberately different opponent (M = 7, ε = 0.5; seeds
   69000–69019), so it doesn't preview the test condition. Its numbers are
   not results.
4. Tag `study3-v1`. The run script refuses a dirty tree or a missing tag
   and records the commit hash. It runs once on seeds 60000+.
5. The results are committed unedited, and every outcome is reported,
   including H1 failing and any "inconclusive".

## Study 3 calibration: result (2026-09-28)

`results/study3/calibration.json`. Reference agents only. The code was
committed (`73706d7`) before the run.

- **M = 5.** The first value tried separates every pair of reference
  agents by more than 2 SE. Their scores on [300, 1000): uniform 0.000,
  always-rock −0.900, 20-move counter −0.476, 50-move counter −0.448,
  full-history counter −0.241.
- **T_w = 858.** Per reference: uniform 0, always-rock 49, 20-move counter
  285, 50-move counter 637, full-history counter 858. T_w > 700, so the
  episode is extended to **1,158 rounds** and the score window is
  [858, 1158).
- **N = 400** per agent: σ_ref = 0.071 and inflation 1 give a required n of
  about 265, which rounds up the grid to 400.

**Caveat, recorded before any frozen-agent run.** T_w is set entirely by
the full-history counter. Against a reactive opponent its play keeps
drifting, so its trailing mean reaches the band only late in the
calibration window. The rule takes the maximum over the references, so this
one slow reference moves every agent's score window to the last 300 rounds.
It is applied as written. The effect is a later, shorter scoring window,
which is conservative, not a bias.

## Study 3 dry run (2026-09-28, before the tag)

The opponent was deliberately different (M = 7, ε = 0.5), with seeds
69000–69019. The pipeline completed and wrote every output.

**Bug caught and fixed.** The H2 outcome label ignored direction: a pair
where the first agent scored *lower* was labeled "above". The label now
names the agent that is above, and a test covers it.

**Disclosure.** The dry run's printed numbers were seen. Its opponent is
from the same family as the test opponent, so they preview direction:
fine-tuning +0.020, RL −0.041, in-context −0.105, with entropies in the same
order (H1 holds there). Every rule was already fixed. Nothing was adjusted
in response, and the test condition (M = 5, ε = 0.1, seeds 60000–60399)
has not been run. Future dry runs should suppress their summary numbers.

## Study 3: results (2026-09-28)

`results/study3/results.json`, committed unedited (`49bbc9f`), from the tag
`study3-v1` (`5421954`). Opponent M = 5, ε = 0.1; 400 seeds per agent
(60000–60399); score window [858, 1158).

| Agent | Score [95% CI] | H3 (vs Nash) | Trailing-M entropy |
|---|---|---|---|
| Fine-tuning | +0.232 [0.229, 0.235] | exploits the opponent | 0.865 |
| Change-aware RL | −0.047 [−0.049, −0.046] | exploited | 0.780 |
| In-context | −0.157 [−0.159, −0.155] | exploited | 0.560 |

- **H2:** every pair is separated by at least the effect of interest.
  Differences: fine-tuning − RL +0.280, RL − in-context +0.109,
  fine-tuning − in-context +0.389. Holm p is at the bootstrap floor, and
  achieved power is about 1.
- **H1: confirmed**, as an ordinal check on n = 3, not a statistical test.
  The entropy order equals the score order, with the in-context agent
  lowest on both.
- **Ranking: changed** against both of study 1's orders. Fine-tuning goes
  from last to first; the in-context agent goes from first on total regret
  to last.
- **Diagnostics:** the median cycle period of the opponent's best response
  is 15 rounds against the in-context agent and 8 against fine-tuning.
  Against RL, 394 of 400 runs have no cycle, because its near-random play
  gives the opponent's tracker little structure to follow.

**What is demonstrated and what is not.**
- *Demonstrated:* fine-tuning wins by a wide, well-powered margin, and it
  is the only agent that exploits the opponent. It is also the only agent
  that both conditions on recent history and changes its mapping during
  play. RL updates every round but has no context input, so it cannot
  represent a contingency. The in-context agent conditions on history, but
  its mapping is frozen and was learned against non-reactive opponents.
- *Not tested:* **why** fine-tuning wins. Its result is *consistent with*
  learning the opponent's reaction rule (the opponent answers the agent's
  last 5 moves, which correlate with the opponent's own recent moves, which
  is what fine-tuning reads). The saved data (summaries only) cannot check
  this. A candidate follow-up would test whether its advantage shrinks as
  M grows, since a longer window should be harder to track with a few
  gradient steps.
- *H1 matched, but it does not explain the headline.* Entropy accounts for
  who is exploited, not for fine-tuning exploiting the opponent. RL's lack
  of cycling is consistent with H1, but for this agent it is not a second
  finding. Near-random play is both high-entropy and unordered, so one
  cause produces both numbers. In general the measures differ (repeating
  rock-paper-scissors has maximal entropy and a 3-round cycle). The saved
  summaries cannot show how closely they track per run.
- *The in-context agent's last place is evidence about transfer,* as the
  hypothesis states. It cannot see its own moves, and it never met a
  reactive opponent in pretraining.
- *The dry run previewed this direction* (disclosed above); nothing was
  adjusted.

## Study 3, bridge condition (b) (2026-09-28, defined before running; exploratory)

**Design.** For rounds [0, 200) the opponent plays study 1's Strategy A,
(0.6, 0.2, 0.2). From round 200 it becomes the study 3 fictitious-play
opponent (M = 5, ε = 0.1). The fictitious-play component observes the
agent's moves from round 0, so its window is full at the switch. Episodes
are 200 + 1,158 = 1,358 rounds. The frozen agents are unchanged; seeds are
61000–61399 (400 per agent).

**Measured per phase, never pooled:**
- **Scripted phase,** rounds [50, 200): mean expected reward and mean
  trailing-5 entropy.
- **Early reactive phase,** rounds [200, 400): mean expected reward. This is
  descriptive: how each agent fares in its first 200 rounds against the
  reactive opponent.
- **Settled reactive phase,** rounds [1058, 1358): study 3's warm-up (858
  rounds) counted from the switch. Mean expected reward, mean trailing-5
  entropy, and the cycle-period diagnostic.
- **Comparison with the primary study 3:** the settled reactive score minus
  the primary score for the same agent, with a bootstrap 95% CI (unpaired,
  10,000 resamples). This shows whether arriving from a scripted regime
  changes long-run play against the reactive opponent.

**Status.** Exploratory: no hypothesis tests, no decision rule, no ranking
claim. The primary design (a) remains the study's result. No recovery time
is computed, since there is no reference level to recover to.

**Process.** The code is committed before the run. The smoke test (4 seeds,
M = 7, ε = 0.5) prints only whether it completed, not its numbers. The
results are committed unedited, with the commit hash recorded.

### Bridge condition (b): results (2026-09-28; exploratory)

`results/study3/bridge.json`, committed unedited (`6f2a761`). The code was
committed (`4de4fa6`) before the run. 400 seeds per agent (61000–61399).
Settled window [1058, 1358).

| Agent | Scripted phase: score / entropy | Early reactive score | Settled reactive score | Settled − primary |
|---|---|---|---|---|
| In-context | +0.383 / 0.05 | −0.161 | −0.152 | +0.005 |
| Change-aware RL | +0.219 / 0.55 | −0.052 | −0.046 | +0.001 |
| Fine-tuning | +0.172 / 0.68 | −0.079 | +0.243 | +0.011 |

- **The two phases reward opposite traits.** The in-context agent is near
  optimal against the scripted opponent (the best possible is 0.4) while
  playing almost deterministically, and is exploited as soon as the
  opponent reacts. The phases are reported separately, as pre-registered.
- **Only fine-tuning improves within the reactive phase,** from −0.079 in
  the first 200 reactive rounds to +0.243 settled. The others are flat.
  This is a demonstrated within-episode change, consistent with it changing
  its mapping during play. The specific mechanism is still untested.
- **Arriving from a scripted regime barely changes long-run play.** All
  settled-minus-primary differences are below the 0.02 effect of interest.
  Their CIs cover only the bridge run's own uncertainty; the primary's CI
  is reported beside them in the results file.

---

# Study 3b: partial observability (draft, 2026-09-28; frozen at tag `study3b-v1`)

The same frozen agents and fictitious-play opponent as study 3 (M = 5,
ε = 0.1, 1,158 rounds, score window [858, 1158)), with observation lag or
noise added. This addresses the outside objection that the setup had "no
latency, no noise". The standing conventions apply.

## Mechanics (`runner.run_episode`; specified by `tests/test_observability.py`)

- **Agent lag k:** the agent receives round t's (own action, opponent move,
  reward) after round t + k. Feedback from the last k rounds never arrives.
- **Opponent lag k:** the opponent's window holds the agent's moves from k
  rounds earlier.
- **Noise q:** with probability q, an observed move is replaced by a draw
  uniform over all three moves, which can come out unchanged, so the actual
  corruption rate is 2q/3. Rewards are never corrupted.
  - The test pins this down: always-rock's expected reward under opponent
    noise matches exact enumeration over all 3⁵ windows.
- **The fine-tuning agent's update under lag is delayed REINFORCE.** It
  stores the input each action was chosen from and applies the gradient to
  that input, at the current weights, when feedback arrives. Without that,
  its update would pair a reward with the wrong input, so it would be
  degraded by the implementation, not by the information loss.
  - With no lag, the new code reproduces study 3's saved trajectories
    exactly for all three agents (tested).

## Order: lag first, then noise

Lag goes first because it is the more direct reading of the objection (no
latency). Noise follows the same design and runs only after the lag
results are recorded.

## Choosing the lag and noise levels (a design check on reference agents only)

A level too weak to move play by the effect of interest (0.02) would spend
a study confirming a null by design. So the levels are chosen on reference
agents, never the three frozen agents, with seeds 70000–70099:
- The five study 3 reference agents are run under **symmetric** lag
  k ∈ {1, 2, 3, 5} and **symmetric** noise q ∈ {0.1, 0.2, 0.3}, plus the
  no-lag, no-noise baseline. Each is scored over [858, 1158).
- **Rule:** k* is the smallest k in {1, 2, 3, 5} at which at least one
  reference agent's score moves by at least 0.02 (closed) *and* by more
  than 2 SE of the difference (strict). q* is chosen the same way from
  {0.1, 0.2, 0.3}.
- **If no level qualifies,** the largest level is used and the condition
  is labeled "minimal perturbation": a check that a small realistic
  perturbation doesn't change conclusions, not a test of an effect.

## Conditions (400 seeds per agent each)

| Condition | Lag study (seeds) | Noise study (seeds) |
|---|---|---|
| **Symmetric** (primary): both sides lagged or noisy | k*, both sides (62000–62399) | q*, both sides (64000–64399) |
| **Agent-only** (secondary): isolates the agent's information, the objection's actual subject | k*, agent side (63000–63399) | q*, agent side (65000–65399) |

**Baseline:** study 3's primary runs (seeds 60000–60399), rerun to get
per-seed scores (the results file stores summaries only). The rerun must
reproduce study 3's per-agent means exactly (asserted).

## Hypotheses (per agent, directional; stated before any run)

Symmetric lag or noise degrades *both* the agent's information and the
opponent's tracking. These can pull in opposite directions for different
agents, so an aggregate null could hide opposite-signed effects. Every test
is therefore **per agent**, never pooled.
- **Fine-tuning:** its score **decreases** under both conditions, because
  the reaction rule it can learn is built from stale or noisy observations.
- **In-context:** its score **increases** (it is less exploited) under the
  symmetric condition, because the opponent's tracker is stale or noisy and
  this is the most exploitable agent. Under agent-only it is **unchanged**
  (|Δ| < 0.02): it cannot exploit this opponent either way.
- **RL:** **unchanged** (|Δ| < 0.02) under both, because near-random play
  gives the opponent little to track, and the agent uses no context.

## Tests

Per agent and condition, Δ = score(condition) − score(baseline). Both are
means over seeds; the CI and p-value come from an unpaired bootstrap with
10,000 resamples. Holm correction across the 3 agents within each
condition. Outcomes, with a 0.02 margin (as in study 3's H2):
- **Holm p < 0.05:** "increases" or "decreases", and by at least or less
  than 0.02.
- **Holm p ≥ 0.05, with the CI strictly inside (−0.02, 0.02):**
  "unchanged within the margin".
- **Otherwise:** inconclusive.

Each directional prediction counts as confirmed only if its category
matches: "increases" or "decreases" with p < 0.05, or "unchanged within the
margin" where no change was predicted.

**If a level falls back to "minimal perturbation"** (no level moved any
reference agent by the bar), a predicted *change* that doesn't appear at
that level is reported as **"not testable at this level"**, never as "the
predicted effect did not occur". This matters most for the in-context
reversal. The same distinction kept study 2's "infeasible as designed"
apart from "no effect". Predictions of *no change* are still evaluated as
written. Added 2026-09-28, while the design check was running and before
its levels were known.

## Process

This draft and the code are committed. The design check (reference agents)
fixes k* and q*, and its results are committed. The dry run uses a
different opponent (M = 7, ε = 0.5) and **prints no numbers**. Then the tag
`study3b-v1`, one run of the lag conditions, results recorded, then the
noise conditions.
