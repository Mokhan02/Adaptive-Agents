# Study 2: early detection (pre-registered)

Frozen at the tag `early-detection-v1`, before any of this study's test
seeds (30000+) exist. Changes after the tag go in **Amendments** at the
bottom.

## Question

Does the in-context agent's internal state give earlier notice of a regime
switch than its output does? This is the proposal's early-warning
motivation. Study 1 (ANALYSIS_PLAN.md) compared *peak timing* and found the
probed state lags the output by about 2.5 rounds. It did not test *first
response*. This study does, on fresh seeds.

**Fixed from study 1, not recalibrated:** the frozen model, the probe site
(layer-1 residual at the final position, latest-move corrected), h = 14,
W = 86, T_w = 31, switch_at = 200, and the A → B test pair. Recalibrating on
new seeds would add a free choice.

## What was seen before this design was frozen (disclosure)

- **Study 1's results,** including the drift-overlay figure on seeds 0–99.
  That figure shows a small bump in the state at rounds 0–3, inside the noise
  band. The bump is why this study exists. Any claim about it must come from
  fresh seeds.
- **Response strengths, read off old calibration switch runs** (seeds
  10000–10099, with controls 10100–10199), during the power analysis for the
  operational test below. The state's typical response is **0.13** of its 5%
  run-level threshold and the output's is **0.33**. These numbers predict
  that the operational test would find "output detects first", largely
  because the state often never crosses its threshold. They were seen before
  the design below was frozen.
- **All power simulations** used only old seeds (10000–10199, 1000–1099) and
  planted signals. No seed from 20000 upward has been generated.

## The operational test: designed, found infeasible, not run

**The design.** Each signal gets a threshold at the 95th percentile of its
per-run maximum displacement in control runs. That fixes a 5% per-run
false-alarm rate for both signals. The detection delay is the first
post-switch crossing, and the test is a paired sign test on the delay
difference, with an effect of interest of 2 rounds. It is implemented and
tested in `early_detection.py`.

**Why it is not run.** Planted-signal power simulations at the measured
response strengths (`results/early_detection/power*.json`) show two defects:

| Seeds | At the effect of interest | Under a planted timing null |
|---|---|---|
| 800 | 80% inconclusive | 82% inconclusive |
| 3,200 | 41% inconclusive | **19% false state-first** |

1. **Underpowered.** At 5% false alarms, both signals often never fire, so
   most delay differences are censoring.
2. **Biased, increasingly with sample size.** `bias.json` locates the bias.
   Among pairs where both signals fire, the output tends to fire first. But
   the output fires less often (48% vs 52% of runs under the null, because
   its heavy-tailed noise puts its threshold at 4.4 SD), and the rule "a
   signal that never fires counts as later" converts that detection-rate gap
   into a signed "state first" lean. Dropping pairs where neither fires is
   neutral. This contradicts the hypothesis that the bias comes from
   selecting on crossings.

**Why switching the primary test is legitimate here.** The switch is driven
by a validity defect (bias under the null, and no power) found in
simulation, before any fresh seed exists. It is not driven by the test
looking unfavorable. It is recorded that the switch happened *after* the
0.13 / 0.33 strengths were seen, and that those strengths predicted
"output detects first". **The operational test's outcome is predicted, not
tested.**

**Scoped conclusion, reported as such.** Practical early warning is
*infeasible as designed* for this probe (the corrected layer-1 state at the
final position), with this detector (displacement at h = 14, per-run 5%
false-alarm threshold), at the measured response strengths. The strengths
are estimates from 100 switch and 100 control runs: state 0.13 [95% CI 0.10, 0.15], output 0.33 [0.26, 0.41] (`strengths.json`, 2,000 bootstrap resamples). Other
detectors (CUSUM, a learned probe) and other probe sites were not tried.
This does not show that the state carries no early information.

## Primary test (amended): normalized mean-curve timing

Compare *when* each signal's across-run mean response reaches a fixed
fraction of its own rise. Normalizing each signal by its own rise removes
the difference in response strength.

**Definitions** (`early_detection.py`; each was fixed after a planted-signal
test exposed a bias in the simpler version):
- **Window:** displacement at h = 14 over [switch_at − W, switch_at + W] =
  [114, 286], for the corrected layer-1 state (standardized with study 1's
  frozen statistics) and the centered output scores.
- **Mean curve:** across runs, then a centered 5-round moving average
  (`SMOOTH`). The same smoothing is applied to both signals.
- **Baseline:** the mean of the smoothed curve over the 86 pre-switch
  rounds.
- **Rise (cross-fitted):** the peak location is taken from one half of the
  runs (even or odd index) and its value from the other half, averaged over
  both directions, minus the baseline. The plain maximum of a noisy curve is
  inflated, more so for the weaker signal. That biased the 50% crossing by
  −0.21 rounds (15% false positives) in planted tests. Cross-fitting removes
  it.
- **Crossing:** the first post-switch round at which the normalized curve is
  at or above the fraction for 3 consecutive rounds (`SUSTAIN`), linearly
  interpolated with the round before. Without the sustain rule, noise near
  the baseline biased the 10% crossing by +1.1 rounds for a weak signal.
- **Statistic:** Δ_f = crossing(output) − crossing(state), in rounds.
  **Positive means the state reaches the fraction first.**
- **Resolution:** both signals are displacements over 14 rounds, which
  limits timing resolution. The bootstrap CI reflects this.

**Fractions and what each answers:**

| Fraction | Question |
|---|---|
| 10% | First response: the early phase, and the closest this test comes to "early notice" |
| 25% | The early rise |
| 50% | Main-rise timing: largely a fresh-seed replication of study 1's peak-timing result, not a test of early warning |

**Inference.**
- Paired bootstrap over runs, 10,000 resamples, recomputing mean curves,
  rises and crossings in each resample. The two-sided p-value is
  2 · min(P(Δ* ≤ 0), P(Δ* ≥ 0)).
- Holm correction across the three fractions.
- The **primary fraction**, chosen by the rule below, gives the headline.
  The other two are reported beside it with equal prominence.

**Outcome categories** (per fraction, margin 2 rounds, both directions):

| Holm-adjusted p | Estimate or CI | Outcome |
|---|---|---|
| < 0.05 | Δ ≥ +2 | state first by ≥ 2 rounds |
| < 0.05 | 0 < Δ < 2 | state first by < 2 rounds |
| < 0.05 | Δ ≤ −2 | output first by ≥ 2 rounds |
| < 0.05 | −2 < Δ < 0 | output first by < 2 rounds |
| ≥ 0.05 | 95% CI within (−2, 2) | no difference of interest |
| ≥ 0.05 | otherwise | inconclusive |

Given study 1, the expected outcomes are "output first" or "no difference
of interest". A state-first result at 10% alongside output-first at 50%
would mean the state stirs first while the output completes its change
first. That would be reported as the two aspects of the response coming
apart, and neither result overrides the primary.

**Negative control:** the layer-2 residual (latest-move corrected,
standardized on these runs' own pre-switch rounds after warm-up) against
the output, at the primary fraction. Layer 2 feeds the output through one
normalization and a linear map, so the expected result is about 0. It
checks that the method doesn't invent a timing difference. It is not a
sensitivity check.

## Primary fraction, seed count and power

**Rule (fixed before the power simulation):** Primary fraction: the lowest of 10/25/50% whose planted-null false-positive rate is <= 6% at every amplitude ratio (0.4, 1, 2.5, measured) and whose power at the measured ratio is >= 80%, at the smallest seed count where any fraction qualifies. If none qualifies at 1600, the primary is 50% at 1600, declared underpowered.

Power simulation (`results/early_detection/timing_power.json`):
- **Noise:** real control traces (old seeds 10100–10199 and 1000–1099) at
  random post-warm-up time offsets (from round 64, where the context is full
  and noise is stationary). The state and output share each offset.
- **Planted response:** the output's empirical mean-response shape (old
  calibration switch runs), used for both signals, so the null has
  identical timing and no bump is built in.
- **Amplitude ratios** (state SNR / output SNR): 0.4, 1, 2.5, and the
  measured ratio (0.19: state SNR 0.62, output SNR 3.32).

Two earlier versions of this simulation were wrong and were fixed before
use:
1. **A shared noise realization.** Resampling 400 runs from 200 traces made
   every experiment share one fixed noise pattern, giving false-positive
   rates of 16–87%.
2. **Non-stationary noise.** Offsets starting before the context window was
   full gave false-positive rates that depended on amplitude.

100 experiments per cell, with 200 bootstrap resamples each (so each rate has an SE of about 2%):

| Seeds | Worst null false-positive rate across ratios (10% / 25% / 50%) | Power at measured ratio | Mean estimate at a true +2 |
|---|---|---|---|
| 400 | 11% / 9% / 8% | 91% / 98% / 100% | +1.16 / +1.98 / +2.02 |
| 800 | 9% / 7% / 8% | 98% / 100% / 100% | +1.16 / +2.03 / +2.10 |
| 1600 | 9% / 7% / 6% | 100% / 100% / 100% | +1.16 / +1.96 / +2.01 |

Null mean estimates are within ±0.15 rounds of 0 in every cell.

**Chosen:** **50% at N = 1,600.** The 10% fraction never qualifies: its worst-case null false-positive rate is 9–11%, and it attenuates a true 2-round lead to about +1.2.

**What this means for the claim.** The pre-registered rule made the primary the *main-rise* fraction. The headline therefore answers the replication question (study 1's timing, on fresh seeds), not the early-warning question. The early-phase fraction (10%) is still reported under Holm correction, with its known defects stated beside it: a false-positive rate up to about 10% under the null, and estimates shrunk toward 0. A "state first" result at 10% would therefore be weak evidence at best. A null there is not strong evidence either, given the shrinkage.

## Seeds

| Use | Seeds |
|---|---|
| Test switch runs | 30000 to 30000 + N − 1 |
| Dry run (pair (0.1, 0.8, 0.1) → (0.1, 0.1, 0.8)) | 39000–39059 |
| Power and diagnostics | old seeds only, as listed above |

No control runs are needed, since the baseline comes from each window's own
pre-switch rounds. The previously planned threshold and held-out control
ranges (20000–21999) belonged to the operational test and are unused.

## Process

1. This document, the code, the planted tests and every power and
   diagnostic file are committed.
2. A dry run on the non-A/B pair (39000+) confirms the pipeline completes
   and writes every output. Its numbers are not results.
3. The commit is tagged `early-detection-v1`.
4. `scripts/early_detection.py run` runs once. It refuses a dirty tree, a
   missing tag or an existing results file, and records the commit hash.
5. `results/early_detection/results.json` is committed unedited, and every
   outcome is reported, including inconclusive ones.

## Amendments

_None._
