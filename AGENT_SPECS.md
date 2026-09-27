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
