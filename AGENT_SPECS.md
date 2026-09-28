# Agent specifications (pre-registered)

Companion to [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md), with the same rules: frozen
before any agent is trained or run on the test opponent, with changes going in
**Amendments** at the bottom. The pretraining sampler that enforces this spec
is `src/regime/pretrain.py`, and its tests are `tests/test_pretrain.py`.

## Test opponent (fixed)

Strategy A = (0.6, 0.2, 0.2) → Strategy B = (0.2, 0.2, 0.6), with the switch at
round 200 of 600. Hard switch (primary); transition ∈ {25, 50, 100, 200}
(exploratory).

## Pretraining distribution (in-context agent)

**Principle:** randomize everything the test fixes, and exclude everything
near what the test uses.

**Held-out strategies.** RPS is invariant under cyclic relabeling of actions
(rock → paper → scissors → rock), so every rotation of A is the same problem
as A. B is itself a rotation of A. The held-out set is therefore the three
rotations: (0.6, 0.2, 0.2), (0.2, 0.6, 0.2) and (0.2, 0.2, 0.6).

Rotations are the only relabelings that preserve the win relation. Swapping
two labels reverses who beats whom, which makes a different game, so the
held-out set has exactly three members.

**Exclusion neighborhood.** No pretraining strategy is within total-variation
distance ε = 0.10 of a held-out strategy, where TV(p, q) = ½ Σᵢ |pᵢ − qᵢ|.
This applies to every regime and to every intermediate blend of a gradual
switch, so a transition can't sweep through the excluded neighborhood. The
check on a blend is exact (`segment_tv`): TV along the blend is convex and
piecewise linear, so its minimum is found at the endpoints or the kinks, not
by sampling points along the path.

The held-out strategies are interior points of the simplex (each coordinate
≥ 0.2), not vertices. For ε < 0.2 the three neighborhoods lie entirely inside
the simplex and do not overlap, since the centers are 0.4 apart. Each one is a
hexagon of area 3ε² in the (p₁, p₂) plane, and the simplex has area ½, so the
excluded share of the uniform Dirichlet(1, 1, 1) distribution is exactly
**18ε²**. `test_excluded_mass_matches_closed_form` checks this.

- Why 0.10: from a 50-move context window, the dominant-action frequency of
  A can only be estimated to within one standard error of √(0.6·0.4/50) ≈ 0.07.
  ε ≈ 1.5 standard errors, so an excluded strategy is one the agent could
  barely tell apart from A using its own context.
- Cost: 18ε² = 18% of the strategy space at ε = 0.10. ε = 0.15 would remove
  40.5%, too much to keep the pretraining distribution broad.

**Everything else is sampled per episode (600 rounds):**

| Parameter | Pretraining | Test |
|---|---|---|
| Regime strategies | Dirichlet(1, 1, 1) with exclusion rejected | A, B |
| Number of switches | uniform {0, 1, 2, 3} | 1 (0 in controls) |
| Switch timing | uniform in [50, 550], at least 30 pure rounds between switches | 200 |
| Abruptness | hard w.p. 0.5, else transition uniform in [1, 200] | 0 (primary) |
| Size of switch | total-variation distance ≥ 0.20 between consecutive regimes | 0.40 |

Episodes with zero switches are included so the agent doesn't learn that a
switch always comes. The agent sees only a rolling window of recent moves,
never the round number, so switch timing can't be learned as a fact about
position.

**Seeds:** pretraining episodes use seeds ≥ 1,000,000, disjoint from the
analysis seeds (0–99) and calibration seeds (10000+).

## Agents

There is one variant of each agent. No agent has a confirmatory and an
exploratory version. The input windows differ between agents because they
play different roles:

- The in-context agent's context window is its only means of adapting. K = 50
  is fixed, because the ε margin above is derived from it.
- The fine-tuning agent adapts through its weights. Its input window only
  supplies local context and is one of its tuned hyperparameters.

### 1. In-context agent (primary)

- **Input:** the last K = 50 opponent moves, one-hot, with learned positional
  embeddings over position *within the window*. The opponent does not react to
  the agent, so the agent's own moves are not input.
- **Model:** causal transformer, 2 layers, d_model = 64, 4 heads.
- **Pretraining objective:** cross-entropy on the opponent's next move, at
  every position.
- **Checkpoint selection:** lowest loss on a validation set drawn from the
  pretraining distribution. The model is never evaluated against A or B
  before the checkpoint is frozen.
- **During play:** frozen weights, and no updates of any kind.
- **Output scores:** expected payoff of each action under the predicted
  next-move distribution. **Policy:** softmax(scores / τ), where τ is tuned
  (default 0.1).
- τ affects gameplay, and therefore regret and the adaptation ranking. It
  cannot affect Tests A and B: the opponent does not react to the agent, and
  the agent's input is only the opponent's moves. So its internal state and
  output scores are identical at any τ.
- **Internal state (primary):** the residual stream after layer 1, at the
  final position. The final layer's residual stream is one linear map away
  from the output scores, so a lead/lag near zero is expected there by
  construction. The middle layer is where representation and output can come
  apart.
- **Internal state (exploratory):** the embeddings, the residual stream after
  layer 2, and the attention patterns.

### 2. Online fine-tuning agent (exploratory)

- **Input:** the last k opponent moves, one-hot and flattened, with
  k ∈ {10, 25, 50} tuned.
- **Model:** MLP with one hidden layer of 32 units, producing 3 logits (the
  output scores). Policy: softmax(logits).
- **Update:** REINFORCE with a running-mean reward baseline, one Adam step
  per round on the realized reward.
- **Internal state:** the hidden activations on the current input. The
  displacement of the flattened weights is also reported, since weight change
  is this agent's adaptation mechanism.

### 3. Change-aware RL agent (exploratory)

- **Model:** recency-weighted action values Q (forgetting factor λ), with a
  surprise statistic (running mean of |r − Q[a]|). The softmax temperature
  rises with surprise measured against its own recent baseline. This is the
  agent's only exploration mechanism; the in-context agent's τ does not apply
  to it.
- **Output scores:** Q. **Internal state:** (Q, surprise, temperature).
- **Caveat recorded in advance:** this agent's internal state contains its
  output scores, so a lead/lag near zero is expected by construction. It is
  in the study for the adaptation ranking, and its Test B is reported but
  not interpreted as evidence either way.

## Hyperparameter tuning (fairness)

Every agent's free hyperparameters get the same budget: a random search of
30 configurations.

| Agent | Tuned hyperparameters |
|---|---|
| In-context | learning rate, training steps, τ |
| Fine-tuning | learning rate, baseline decay, input window k |
| RL | λ, base temperature, surprise gain |

Configurations are scored by mean oracle-normalized excess regret against
opponents drawn from `sample_pretraining_opponent`. That sampler excludes the
neighborhoods around **all three** held-out strategies, so no agent is tuned on
A, B or the third rotation, directly or through near-duplicates.
Tuning the baselines on the test strategies while holding them out of the
in-context agent's pretraining would tilt the adaptation ranking toward the
baselines.

## Amendments

### 2026-09-27: remove the latest-move component from the in-context agent's state

Made before the in-context checkpoint was frozen, and before any model had
been run against A or B.

**Change.** The in-context agent's internal state, both the primary site
(residual stream after layer 1, final position) and the exploratory residual
sites, is now the raw residual stream minus the mean residual stream for the
latest opponent move:

`s_t = r_t − μ[m_t]`, where `m_t` is the opponent's latest move and
`μ[k]` is the mean final-position residual stream over full windows ending
in move k.

- **μ is estimated only on pretraining-distribution opponents:** 200 episodes
  from `sample_pretraining_opponent`, seeds 4,000,000+, rounds ≥ 50, once per
  frozen checkpoint (`estimate_move_offsets`). A and B are never used. This
  is the same rule as every other calibrated quantity.
- The per-dimension standardization in ANALYSIS_PLAN.md is applied after
  this correction.
- The raw, uncorrected state is still logged and reported as exploratory.
- Output scores (the behavior signal) are unchanged.

**Reason.** The final position is where the newest move enters, so the raw
state there is dominated by that move's embedding. It changes at random every
round whatever the regime. On pretraining-style opponents (never A or B), the
latest move explained 98% of the raw layer-1 state's variance in an untrained
network and 94% in the step-1,000 checkpoint. Detection of a switch
(`scripts/check_drift_signal.py`, h = 20, chance 5%) was:

| h = 20 detection | Raw state | Latest move removed | Output scores |
|---|---|---|---|
| Untrained network | 37% | 60% | 15% |
| Trained, step 1,000 | 25% | 47% | 79% |

**Alternatives considered.**
- **Averaging the state over all window positions:** rejected. It discards
  position-specific, context-weighted information, which leaves something
  close to the bag-of-moves summary the counting agent already computes.
- **Using layer 1's attention output:** rejected. The final position still
  attends to itself, so the latest move re-enters in a less interpretable
  form.
- The correction chosen removes the confound in exactly the form it was
  measured.

**Not conditional on results.** This definition holds whatever the finished
checkpoint's diagnostic numbers turn out to be. Those numbers are reported,
not used to choose between definitions.

### 2026-09-27: correction to the detection figures above, and frozen-checkpoint diagnostic

**Correction.** The "latest move removed" figures in the previous amendment
(60% untrained, 47% at step 1,000) came from a preliminary check that
estimated the per-move means *in-sample*, from the diagnostic's own control
runs (60 pairs). With the pre-registered procedure (means from separate
pretraining-distribution episodes, seeds 4,000,000+, 100 pairs), the
untrained network's corrected detection is 39%. The decision was fixed
unconditionally and does not change; only the cited reasons were overstated.

**Frozen checkpoint.** Saved as `models/icl_frozen.pt`: step 17,000 of
20,000, validation loss 0.0624 nats above the oracle floor, weights SHA-256
`e0462ec6…c337ce`, with its move offsets bundled. The training log is
`models/icl_frozen.log.jsonl`. It trained in 52 minutes on an Intel Core
i3-1000NG4 (CPU, 3 threads).

Diagnostic (`scripts/check_drift_signal.py`, pretraining-style opponents
only, 100 pairs, chance 5%). Reported as-is, not used to revisit any
definition:

| Detection | h = 1 | h = 10 | h = 20 |
|---|---|---|---|
| Frozen: state, raw | 17% | 31% | 58% |
| Frozen: state, corrected (primary) | 4% | 41% | 60% |
| Frozen: output scores | 42% | 73% | 94% |
| Untrained: state, raw | 3% | 6% | 38% |
| Untrained: state, corrected | 1% | 36% | 39% |
| Untrained: output scores | 6% | 10% | 15% |

### 2026-09-27: tuning protocol, fixed before any configuration runs

**The frozen checkpoint above is provisional.** Learning rate and training
steps are among the in-context agent's tuned hyperparameters (see
Hyperparameter tuning), so the final model is the sweep's winner.
`models/icl_frozen.pt` is the default configuration's candidate, and its
diagnostic table describes that configuration only. No model has been run
against A or B, so re-freezing does not compromise the hold-out. The final
model is frozen with `scripts/freeze_icl.py` in the same way.

**Order of operations.**
1. Run the sweep (below).
2. Freeze the winning configuration of each agent.
3. Run the calibration seeds (10000+) on A → B, which set T_w, h, W and L
   per ANALYSIS_PLAN.md.
4. Report the drift diagnostic at each agent's assigned h.
5. Run the analysis seeds (0–99).

h depends on recovery time, which depends on tuned hyperparameters such as
the in-context temperature, so it cannot be fixed before step 3.

**Tuning episodes.** 200 episodes, seeds 5,000,000–5,000,199, disjoint from
the analysis (0–99), calibration (10000+), pretraining (1,000,000+),
validation (2,000,000+), diagnostic (3,000,000+) and move-offset (4,000,000+)
seeds. Each episode has one switch at round 200 of 600, matching the test
structure:
- Strategies p → q are drawn by the pretraining rules: outside the held-out
  neighborhoods, TV(p, q) ≥ 0.20, and exact path check for gradual switches.
- Hard switch with probability 0.5; otherwise the transition is uniform in
  [1, 200].

Every configuration of every agent plays the same 200 episodes with the same
seeds (common random numbers).

**Score.** Mean oracle-normalized excess regret (`metrics.excess_regret`,
horizon 100) over the 200 episodes. Lower is better. It is the same for all
three agents.

**Search.** 30 configurations per agent, drawn from the ranges below with
`numpy.random.default_rng(6_000_000 + agent_index)`, where agent_index is
0 = in-context, 1 = fine-tuning, 2 = RL. "log" means log-uniform.

| Agent | Hyperparameter | Range |
|---|---|---|
| In-context | learning rate | log [1e-4, 1e-3] |
| | training steps | {2500, 5000, 10000, 20000} |
| | τ | log [0.02, 0.5] |
| Fine-tuning | learning rate | log [1e-3, 1e-1] |
| | baseline decay | [0.8, 0.99] |
| | input window k | {10, 25, 50} |
| RL | forgetting factor λ | [0.8, 0.99] |
| | base temperature | log [0.02, 0.5] |
| | surprise gain | [0, 5] |

Ties are broken by lower configuration index. In-context configurations
each require their own pretraining run (same seeds and data as the default
run, differing only in learning rate and steps).

**RL agent details fixed here** (they were unspecified): Q starts at 0; only
the chosen action's value is updated, `Q[a] += (1 − λ)(r − Q[a])`. Surprise
is `|r − Q[a]|` before the update, tracked by a fast EMA with rate (1 − λ)
and a slow EMA with rate (1 − λ)/10. The temperature is
`τ = τ₀ · (1 + g · max(0, fast/slow − 1))`.

**Fine-tuning agent details fixed here:** the hidden layer uses tanh, and
windows shorter than k (early rounds) are zero-padded. The loss is
`−(r − b) log π(a | x)`, with `b ← d·b + (1 − d)·r` updated after the step.

### 2026-09-27: sweep results and frozen winners

All three sweeps ran to the protocol above: 30 configurations each, on the
same 200 tuning episodes. No agent has been run against A or B. Winners
(lowest score, ties to lower index) are frozen in
`results/tuning/winners.json`:

| Agent | Winner | Tuning score | Next best |
|---|---|---|---|
| In-context | #22: lr 4.6e-4, 20,000 steps, τ 0.035 | 11.88 ± 3.19 | 11.93 (#9), 12.20 (#18) |
| Change-aware RL | #3: λ 0.81, τ₀ 0.30, surprise gain 3.9 | 14.97 ± 3.16 | 15.23 (#6), 15.29 (#1) |
| Fine-tuning | #5: lr 1.0e-3, baseline decay 0.97, k = 10 | 53.07 ± 4.72 | 65.81 (#18), 70.32 (#15) |

Tuning scores are best-of-30 and optimistically biased; they are not the
adaptation ranking. That comes from the analysis seeds (ANALYSIS_PLAN.md).
Observations recorded now, before any test data exists:

- **Near-ties:** the in-context and RL winners are each within 0.1 SE of
  the runner-up. The rule was applied as written.
- **RL surprise gain barely matters:** the top three span gains of 0.6-3.9,
  and the forgetting factor (0.81-0.89) is what separates good
  configurations. The surprise-scaled exploration contributes little in
  this setting.
- **Fine-tuning winner at the edge of its range:** lr 1.0e-3 is the lower
  bound of the range, and the top two are both at the low end. The range may
  understate this agent. The confirmatory ranking uses the frozen winner; a
  lower-lr sensitivity check may be reported, labeled exploratory.
- **In-context: longer training helps gameplay** even though validation
  loss is flat past ~2,500 steps. All of the top four trained for 20,000
  steps, with τ 0.03-0.05.

**The frozen in-context model** is `models/icl_frozen.pt`, replacing the
provisional default-configuration checkpoint: step 17,000 of 20,000,
validation loss 0.0622 above the oracle floor, weights SHA-256
`fdb5ad02…d4a4`, trained on an NVIDIA A10 (Lambda Cloud). Drift diagnostic
(pretraining-style opponents only, 100 pairs, chance 5%):

| Detection | h = 1 | h = 10 | h = 20 |
|---|---|---|---|
| State, corrected (primary) | 7% | 32% | 66% |
| State, raw | 6% | 27% | 68% |
| Output scores | 57% | 74% | 88% |

The untrained baseline for comparison is 39% at h = 20 (corrected state).

**Incidents during the in-context sweep, none affecting the results:**
- The instance's software stack differed from the development machine:
  Python 3.10, an older numpy lacking `Generator.spawn`, and torch 2.6,
  whose `torch.load` rejects numpy scalars by default. Each crashed the
  sweep, and each was fixed in code. numpy was upgraded to 1.26.4, the same
  version used for the other two sweeps.
- Eight 20,000-step runs had trained under the old numpy. Their saved
  learning rates differed from the configurations in the last bits
  (numpy's `exp` and `log` changed), so the resume guard refused them.
  They were deleted and retrained under numpy 1.26.4.
- For a period, two sweep processes ran at once. They never trained the
  same configuration at the same time; the second process only rescored
  finished runs, reproducing their scores exactly. The 15 duplicate result
  lines were identical to the originals and were removed.

**Reproducibility:** results are reproducible to within floating-point
noise across machines, not bit-for-bit. numpy's `exp` and `log` can differ
in the last bit across versions and CPUs.
