# Probe capacity and answer identity

**Status:** EXPLORATORY, not preregistered. Source: `scripts/s16_probe_capacity.py`, which
writes `results/s16_probe_capacity.json`. This doc follows up root cause 5 of
`docs/representation_rootcause.md` (s14): is the failure to decode the outcome a limitation
of linear PCA-128 probes, or of the stored state itself?

## Verdict

1. **Richer probes do not change the s14 conclusion.**
   - The settings tested were 3 layers × 6 model families × PCA 32/128/512 × pre/post/pair
     features × raw/within-trace-demeaned: 108 rows per target.
   - In none of them does the stored boundary state decode:
     - the accuracy level within a trace (best of 108 rows is 0.07, against a ceiling of 0.96);
     - the signed change in accuracy (best 0.04, ceiling 0.52);
     - the debiased anchor effect (best 0.08, ceiling 0.40).
   - Nonlinear models (gradient boosting, two MLPs) are never better than ridge by more than
     bootstrap noise, and are often worse.
   - Within-trace demeaning does not help these targets.
   - The pair design [h_i, h_{i+1}, h_{i+1} − h_i] does not decode signed Δacc. All three
     nonlinear pair models give within-trace values from −0.02 to 0.01, and every CI
     includes 0.
   - The positive control (entropy) behaves as in s14: 0.44–0.60 within a trace.
2. **Answer identity looks far more decodable than correctness, but this is an artefact of
   time and uncertainty, not evidence that the state encodes which answer it is heading for.**
   - Within a trace, `p_own` (agreement with the trace's own final answer) decodes at
     0.52–0.71, against −0.02 to 0.07 for `acc_level`. The paired difference has CIs far
     above 0.
   - But position alone (sentence index and prefix length) predicts `p_own` within a trace
     at **0.79**, better than any activation probe.
   - Controlling for position, the activation probe's partial correlation with `p_own` is
     0.07 [0.02, 0.13] at layer 32.
   - Controlling for position and answer entropy, it is 0.02 [−0.02, 0.06].
   - Changes in answer identity (`dp_own`, `dp_modal`) are as undecodable as Δacc: at most
     0.05 within a trace, against a ceiling of about 0.5.
3. **Why `acc_level` looks undecodable: sign cancellation, not missing information.**
   - In a correct trace, accuracy equals `p_own` (difference exactly 0). In an incorrect
     trace, accuracy is almost `1 − p_own` (Spearman −0.96).
   - Both follow the same "commitment grows over the trace" signal, with opposite signs.
   - The `acc_level` probe is therefore strongly *negative* within correct traces (−0.35)
     and *positive* within incorrect ones (+0.42). The average (0.04) looks like zero.
   - Given the trace's correctness bit, the decoded `p_own` recovers accuracy within a trace
     at 0.65 [0.59, 0.69]. This is an oracle, not a probe.
   - The state does not supply that bit. Telling a problem's correct trace from its
     incorrect trace gives a row-level AUC of 0.49. The correct trace is scored higher in
     13 of 20 problems (layer 32), which is chance level.

**Bottom line.** The boundary residual stream carries *how committed or uncertain* the model
is, and that commitment largely tracks position in the trace. It carries neither *whether
the trace's answer is right* nor *how much the next sentence moves the outcome*. This holds
for linear and nonlinear probes alike, so root cause 5 is answered: the limitation is in the
representation (as read at one boundary token, at 20 problems), not in the linear probe.

## Protocol

- **Data.** The same data as s14: 6,434 sentences (with forks i and i+1), 40 traces and
  20 problems. Targets come from 100 released resamples per fork
  (`derived/rollouts_v2.parquet`).
- **States.** The stored residual stream at the boundary token, at layers 24, 32 and 40,
  as float32.
  - `pre` = h_i and `post` = h_{i+1}.
  - `pair` = [h_i, h_{i+1}, h_{i+1} − h_i], with each block reduced to PCA-k.
- **Leave-one-problem-out (20 folds).** In each fold, one StandardScaler and one PCA-512
  are fit on every stored boundary state of the training problems.
  - Pre and post states are both projected into that basis.
  - PCA-32 and PCA-128 are its leading components, so the PCA dimensions are nested.
  - Target scalers and models are fit on the training problems only.
- **Variants.**
  - `raw`: as in s14.
  - `demean`: each trajectory's mean scaled activation is subtracted before PCA. Each
    trajectory's mean target is subtracted from the training targets.
  - With demeaning, the model can learn only within-trace variation, and problem or trace
    identity is removed.
  - Demeaning the test trace's features uses that trace's own activations, but no labels.
- **Models.**
  - RidgeCV on PCA-32/128/512 (alphas 1e-1 to 1e7).
  - HistGradientBoosting on PCA-32 and PCA-128: 150 iterations, learning rate 0.1, 15
    leaves, min leaf 40, early stopping. Pre and pair features only, for runtime.
  - MLP (256), alpha 1, and MLP (512, 256), alpha 10, on whitened PCA-128, with early
    stopping.
  - Ridge on all 5,120 dimensions was dropped: it took about 75 s per fit, or about 2.5 h
    in total. PCA-512 keeps 83% of the variance.
  - Early stopping in HGB and the MLPs uses a random 10% of *training* rows. This cannot
    leak test data, but may under-regularise against problem identity.
- **Targets.** The s14 targets `acc_level`, `entropy`, `tv_debiased` and `signed_dacc`,
  plus four new ones:

| Target | Definition |
|---|---|
| `p_own` | share of the resamples at fork i whose answer equals the trace's own final answer |
| `p_modal` | modal-answer share at fork i |
| `dp_own` | `p_own` at fork i+1 minus `p_own` at fork i |
| `dp_modal` | `p_modal` at fork i+1 minus `p_modal` at fork i |

- **The own answer.**
  - It is `normalize_v2(trajectories.normalized_final_answer)`, that is, the trace's own
    eventual final answer, not the modal answer of the last fork.
  - It matches the rollout `cat` vocabulary for all 40 traces.
  - It differs from the last fork's modal answer for one trace: 4682:incorrect, own answer
    20, last-fork mode 19 (56%).
- **Ceilings.** Split-half (even vs odd `sample_idx`), Spearman–Brown projected to n = 100,
  then the square root. `p_modal` is computed within each half.
- **Metrics.**
  - Overall Spearman (raw variant only).
  - Mean within-trajectory Spearman (`within`), also split by correct and incorrect traces.
- **Uncertainty.**
  - 1,000-replicate problem-cluster bootstrap with shared draws, so paired differences
    are valid.
  - It covers every layer-32 row, the paired comparisons, and the controls.
  - Layer 32 was fixed in advance as the headline layer. All layers are reported.
- **Runtime.** 60 min for sections A–D. 1 min for section E (controls), added after the
  first run and merged with `--extra-only`; it uses the same folds and the same data.

## 1. Ceilings

| Target | Overall | Within | Within, correct traces | Within, incorrect traces |
|---|---|---|---|---|
| acc_level | 0.99 | 0.96 | 0.95 | 0.96 |
| entropy | 0.98 | 0.93 | 0.95 | 0.92 |
| tv_debiased | 0.39 | 0.40 | 0.37 | 0.42 |
| signed_dacc | 0.51 | 0.52 | 0.51 | 0.53 |
| p_own | 0.99 | 0.96 | 0.95 | 0.97 |
| p_modal | 0.98 | 0.94 | 0.95 | 0.93 |
| dp_own | 0.52 | 0.53 | 0.51 | 0.55 |
| dp_modal | 0.49 | 0.51 | 0.48 | 0.53 |

How the targets relate:

| Relation | Spearman |
|---|---|
| `p_own` vs `acc_level`, all rows | 0.20 |
| `p_own` vs `acc_level`, incorrect traces | −0.96 |
| `p_own` − `acc_level`, correct traces | identical (difference 0) |
| `p_modal` vs `entropy` | −0.97 |
| `dp_own` vs `signed_dacc` | −0.12 |

## 2. Range over all 108 settings per target (3 layers × models × features × variants)

The maximum is a selection-inflated upper bound.

| Target | Within ceiling | Within: min / median / max | Overall ceiling | Overall (raw): min / median / max |
|---|---|---|---|---|
| acc_level | 0.96 | −0.02 / 0.02 / **0.07** | 0.99 | 0.04 / 0.10 / 0.17 |
| signed_dacc | 0.52 | −0.03 / 0.01 / **0.04** | 0.51 | −0.03 / 0.00 / 0.02 |
| tv_debiased | 0.40 | −0.01 / 0.04 / **0.08** | 0.39 | −0.01 / 0.04 / 0.08 |
| dp_own | 0.53 | −0.03 / 0.01 / 0.04 | 0.52 | −0.02 / 0.01 / 0.02 |
| dp_modal | 0.51 | −0.02 / 0.02 / 0.05 | 0.49 | −0.02 / 0.01 / 0.04 |
| entropy (control) | 0.93 | 0.44 / 0.54 / 0.60 | 0.98 | 0.39 / 0.46 / 0.50 |
| p_modal | 0.94 | 0.39 / 0.49 / 0.54 | 0.98 | 0.34 / 0.42 / 0.46 |
| p_own | 0.96 | 0.52 / 0.63 / 0.71 | 0.99 | 0.36 / 0.43 / 0.48 |

The full per-layer, per-model tables are in `results/s16_probe_capacity.json`
(`predictability`).

### Within-trace Spearman, raw variant, every model and feature set

Layers are shown as 24 / 32 / 40.

| Model | Features | acc_level | signed_dacc | tv_debiased | entropy | p_own |
|---|---|---|---|---|---|---|
| ridge PCA-32 | pre | −.01/.01/.01 | −.02/.01/.00 | .04/.05/.05 | .49/.52/.50 | .57/.60/.61 |
| ridge PCA-32 | pair | −.01/.02/.02 | .00/.03/.04 | .06/.07/.07 | .52/.55/.53 | .59/.63/.62 |
| ridge PCA-128 | pre | .02/.04/.02 | .00/.01/−.01 | .04/.04/.05 | .53/.54/.51 | .62/.65/.62 |
| ridge PCA-128 | post | .02/.04/.02 | .02/.02/.03 | .06/.08/.07 | .54/.54/.50 | .63/.65/.62 |
| ridge PCA-128 | pair | .02/.05/.03 | .02/.02/.03 | .07/.07/.06 | .55/.56/.53 | .64/.67/.64 |
| ridge PCA-512 | pre | .05/.04/.07 | .00/.01/.00 | .04/.04/.05 | .53/.55/.52 | .63/.64/.60 |
| HGB PCA-32 | pair | .01/.04/.03 | −.01/.00/.02 | .03/.03/.03 | .50/.51/.51 | .55/.56/.59 |
| HGB PCA-128 | pre | .03/.05/.02 | .00/.00/.00 | .01/.01/−.01 | .50/.50/.47 | .55/.56/.54 |
| HGB PCA-128 | pair | .02/.05/.02 | .01/−.01/.01 | .00/.04/.04 | .52/.52/.51 | .57/.57/.57 |
| MLP 256 | pre | .05/.03/.05 | .00/.01/.00 | .03/.02/.00 | .51/.51/.47 | .57/.58/.54 |
| MLP 256 | pair | .04/.05/.03 | .01/.00/−.02 | .02/.05/.00 | .48/.49/.44 | .54/.55/.52 |
| MLP 512×256 | pre | .06/.05/.04 | .00/.00/.00 | .03/.02/.04 | .53/.53/.48 | .63/.65/.60 |
| MLP 512×256 | post | .04/.06/.04 | .01/.00/.01 | .01/.06/.05 | .53/.54/.50 | .63/.64/.61 |
| MLP 512×256 | pair | .05/.07/.05 | .01/−.02/.00 | .04/.05/.05 | .54/.55/.50 | .63/.65/.60 |

Rows not shown (ridge PCA-32 post, ridge PCA-512 post, HGB PCA-32 pre, MLP 256 post) are
within ±0.03 of their neighbours. All are in the JSON.

### Within-trace Spearman, demeaned variant, layer 32

| Model | Features | acc_level | signed_dacc | tv_debiased | entropy | p_own |
|---|---|---|---|---|---|---|
| ridge PCA-128 | pre | 0.02 [−0.04, 0.08] | 0.01 [−0.01, 0.04] | 0.05 [0.02, 0.08] | 0.58 [0.50, 0.65] | 0.70 [0.66, 0.74] |
| ridge PCA-512 | pre | 0.03 | 0.01 | 0.05 | 0.59 | 0.70 |
| HGB PCA-128 | pair | 0.03 [−0.02, 0.09] | 0.01 [−0.01, 0.03] | 0.03 [−0.00, 0.05] | 0.56 [0.49, 0.63] | 0.64 [0.59, 0.68] |
| MLP 512×256 | pair | 0.03 [−0.04, 0.10] | −0.00 [−0.02, 0.01] | 0.05 [0.03, 0.07] | 0.59 [0.51, 0.66] | 0.69 [0.65, 0.74] |

- Demeaning raises the median within-trace value slightly for entropy (0.52 → 0.56) and
  `p_own` (0.60 → 0.66).
- It lowers it for `acc_level` (0.04 → 0.01).
- Removing problem identity therefore does not unlock correctness.

### Headline layer 32, raw variant, with 95% problem-cluster bootstrap CIs

| Target (overall / within ceiling) | ridge PCA-128 pre (s14 replica) | MLP 512×256 pair | HGB PCA-128 pair |
|---|---|---|---|
| acc_level, overall (0.99) | 0.16 [0.02, 0.30] | 0.16 [0.01, 0.30] | 0.11 [−0.01, 0.24] |
| acc_level, within (0.96) | 0.04 [−0.03, 0.11] | 0.07 [−0.00, 0.15] | 0.05 [−0.01, 0.11] |
| signed_dacc, overall (0.51) | 0.00 [−0.01, 0.02] | −0.02 [−0.05, 0.01] | −0.01 [−0.03, 0.02] |
| signed_dacc, within (0.52) | 0.01 [−0.01, 0.04] | −0.02 [−0.05, 0.01] | −0.01 [−0.04, 0.02] |
| tv_debiased, overall (0.39) | 0.05 [0.01, 0.08] | 0.06 [0.03, 0.08] | 0.05 [0.02, 0.07] |
| tv_debiased, within (0.40) | 0.04 [0.01, 0.07] | 0.05 [0.03, 0.08] | 0.04 [0.01, 0.07] |
| entropy, within (0.93) | 0.54 [0.47, 0.61] | 0.55 [0.48, 0.62] | 0.52 [0.45, 0.58] |
| p_own, within (0.96) | 0.65 [0.60, 0.71] | 0.65 [0.59, 0.71] | 0.57 [0.51, 0.62] |
| dp_own, within (0.53) | 0.03 [0.01, 0.04] | 0.01 [−0.01, 0.03] | 0.01 [−0.01, 0.03] |

### Paired comparison against the s14 baseline (layer 32, within trace)

Each of the 35 other layer-32 settings was compared with raw ridge PCA-128 on `pre`, for
every target.

- **Largest gain on the failed targets:**
  - `acc_level`: +0.04 (raw MLP 512×256 pair, CI [0.01, 0.07]);
  - `signed_dacc`: +0.02 (no CI excludes 0);
  - `tv_debiased`: +0.03.
- **Other gains are small.** Settings whose CI excludes 0 gain at most +0.055, mostly on
  entropy, `p_own` and `p_modal` (demeaned ridge or MLP).
- **Losses are as common as gains.** Many nonlinear settings are significantly *worse*
  than the baseline: 13 of 35 on `p_own`.

## 3. Answer identity versus correctness

### Raw difference (layer 32)

The contrast is `p_own` − `acc_level`, within a trace.

| Model | Raw | Demeaned |
|---|---|---|
| ridge PCA-128 pre | +0.62 [0.53, 0.70] | +0.68 [0.61, 0.74] |
| MLP 512×256 pair | +0.57 [0.48, 0.66] | +0.66 [0.59, 0.73] |
| HGB PCA-128 pair | +0.52 [0.44, 0.61] | +0.61 [0.53, 0.67] |

- Every layer-32 setting, raw or demeaned, has a CI excluding 0.
- The overall (cross-problem) difference is +0.27 to +0.40, again with every CI above 0.
- On face value, answer identity is far more decodable than correctness. The controls
  below show why that is misleading.

### Controls (section E, ridge PCA-128 `pre`, raw)

**Position-only baseline** (s14 features: index, log index, log prefix length; same leave-
one-problem-out protocol):

| Target | Within, position | Within, activations L32 | Within, activations + position L32 |
|---|---|---|---|
| p_own | **0.79** | 0.65 | 0.71 |
| p_modal | **0.65** | 0.49 | 0.53 |
| entropy | **0.70** | 0.54 | 0.57 |
| acc_level | 0.03 (correct −0.72, incorrect +0.79) | 0.04 (correct −0.35, incorrect +0.42) | 0.07 |
| signed_dacc | 0.01 | 0.01 | 0.01 |
| tv_debiased | 0.05 | 0.04 | 0.04 |
| dp_own | 0.01 | 0.03 | 0.03 |

**Partial within-trace Spearman** of the decoded `p_own` (activations) with the true
`p_own`:

| Controls | L24 | L32 | L40 |
|---|---|---|---|
| position | 0.04 [−0.01, 0.10] | 0.07 [0.02, 0.13] | 0.06 [0.01, 0.11] |
| position + entropy | −0.01 [−0.05, 0.03] | 0.02 [−0.02, 0.06] | 0.02 [−0.02, 0.05] |
| position + p_modal | 0.01 [−0.03, 0.05] | 0.01 [−0.02, 0.04] | 0.02 [−0.01, 0.05] |
| position + entropy + p_modal | 0.02 [−0.02, 0.06] | 0.02 [−0.01, 0.05] | 0.03 [−0.00, 0.05] |

- **Position does carry information beyond uncertainty.** For comparison, the position
  probe's partial correlation with `p_own`, controlling for entropy, is 0.32.
- **The activation probe adds almost nothing.** Once position and uncertainty are known,
  it adds about 0.02 about the trace's own path.

**Sign cancellation in `acc_level`:**

| Check | L24 | L32 | L40 |
|---|---|---|---|
| Within-trace Spearman of the `acc_level` probe with the `p_own` probe | −0.48 | −0.43 | −0.41 |

- **Oracle-sign accuracy.** This takes the decoded `p_own` if the trace is correct, and
  its negative otherwise. Within a trace it reaches 0.61 / 0.65 / 0.61, with layer-32 CI
  [0.59, 0.69].
- **What it shows.** Correctness is recoverable only if one already knows whether the
  trace ends right.

**Can the state tell whether its trace is right?** This is a logistic probe on PCA-128
`pre` for "this is the correct trace", leave-one-problem-out.

| Layer | Row AUC | Mean within-problem AUC | Correct trace scored higher (all / late half) |
|---|---|---|---|
| 24 | 0.49 | 0.55 | 11/20 / 13/20 |
| 32 | 0.49 | 0.54 | 13/20 / 13/20 |
| 40 | 0.45 | 0.50 | 12/20 / 11/20 |

All of these are at chance: 13 of 20 gives a two-sided binomial p of 0.26.

## Interpretation and caveats

- **What the state carries.** Within a trace, the decodable content of the boundary state
  is a *commitment / uncertainty* axis.
  - The answer distribution sharpens toward the trace's own answer as reasoning proceeds.
  - Position tracks this better than the state does.
  - Across problems, the state adds problem-level uncertainty (overall entropy 0.48,
    against 0.33 for position in s14).
- **What it does not carry.** It does not, at the resolution tested:
  - say whether the trace is heading for the correct answer;
  - predict which sentence changes the outcome.
- **The effect labels are noisy.** `signed_dacc`, `dp_own` and `tv_debiased` have
  ceilings of only 0.4–0.5. Even so, every probe sits at under 15% of ceiling, with CIs
  that mostly include 0, so label noise alone does not explain the null.
- **Limitations:**
  - 20 problems;
  - one boundary token per sentence;
  - the stored layers only (24/32/40 here);
  - PCA ≤ 512;
  - random-row early stopping;
  - HGB was not run on `post` features;
  - no ridge on all 5,120 dimensions.
- **Selection.** The headline layer (32) and the comparison baseline were fixed before
  the results were seen. The controls in section E were added *after* seeing the large
  `p_own` result, as a check of the alternative explanations of position and uncertainty.
  They are exploratory in the same sense.
- **Implication.** The forced-answer readout proposed in s14, which reads the model's own
  answer distribution at each boundary, remains the most direct route. A residual-stream
  "answer-identity probe" from one boundary token is not a substitute: what it reads is
  mostly position and uncertainty.
