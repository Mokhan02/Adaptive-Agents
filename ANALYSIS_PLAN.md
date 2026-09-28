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
