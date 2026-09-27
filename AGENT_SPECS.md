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

**Exclusion neighborhood.** No pretraining strategy is within total-variation
distance ε = 0.10 of a held-out strategy. This applies to every regime and to
every intermediate blend of a gradual switch, so a transition can't sweep
through the excluded neighborhood.

- Why 0.10: from a 50-move context window, the dominant-action frequency of
  A can only be estimated to within one standard error of √(0.6·0.4/50) ≈ 0.07.
  ε ≈ 1.5 standard errors, so an excluded strategy is one the agent could
  barely tell apart from A using its own context.
- Cost: this removes 18% of the strategy space (the space of all three-way
  mixed strategies). ε = 0.15 would remove 40%, too much to keep the
  pretraining distribution broad.

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
  next-move distribution. **Policy:** softmax(scores / 0.1).
- **Internal state (primary):** the residual stream after layer 1, at the
  final position. The final layer's residual stream is one linear map away
  from the output scores, so a lead/lag near zero is expected there by
  construction. The middle layer is where representation and output can come
  apart.
- **Internal state (exploratory):** the embeddings, the residual stream after
  layer 2, and the attention patterns.

### 2. Online fine-tuning agent (exploratory)

- **Input:** the last 10 opponent moves, one-hot and flattened.
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
  rises with surprise measured against its own recent baseline.
- **Output scores:** Q. **Internal state:** (Q, surprise, temperature).
- **Caveat recorded in advance:** this agent's internal state contains its
  output scores, so a lead/lag near zero is expected by construction. It is
  in the study for the adaptation ranking, and its Test B is reported but
  not interpreted as evidence either way.

## Hyperparameter tuning (fairness)

Every agent's free hyperparameters get the same budget: a random search of
30 configurations. For the fine-tuning agent that means learning rate and
baseline decay; for the RL agent, λ, the base temperature and the surprise
gain; for the in-context agent, learning rate and training steps.
Configurations are scored by mean oracle-normalized excess regret against
opponents drawn from the **pretraining distribution**, never against A or B.
Tuning the baselines on the test strategies while holding them out of the
in-context agent's pretraining would tilt the adaptation ranking toward the
baselines.

## Amendments

_None._
