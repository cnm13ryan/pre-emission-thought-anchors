# Root cause: why the stored representations do not predict thought anchors

**Status:** EXPLORATORY, not preregistered. Source: `scripts/s14_representation_rootcause.py`,
which writes `results/s14_representation_rootcause.json` and `results/s14_oof_predictions.parquet`.

## Data and protocol

- **Data:** 40 MVP traces (20 problems). 6,434 sentences have both fork i and fork i+1.
  Each fork has 100 released resamples.
- **Stored state:** the residual stream at the token just *before* each sentence. The
  analysis uses layers 8, 16, 24, 32, 40 and 48.
- **Models:** leave-one-problem-out (20 folds). Scaling, PCA-128 and RidgeCV are fit on the
  training problems only.
- **Metrics:** Spearman overall, and the mean within-trajectory Spearman.
- **Representations:**

| Name | What it is |
|---|---|
| `pre` | h_i, the state before S_i (what the earlier probe used) |
| `post` | h_{i+1}, the state after S_i |
| `delta` | post − pre |
| `pre+post` | both states |
| `text_S_i` | MiniLM embedding of the realised sentence |
| `position` | sentence index, log index and log prefix length |

- **Targets:**

| Name | Definition |
|---|---|
| `acc_level` | accuracy at fork i |
| `entropy` | entropy of the answer distribution at fork i |
| `floor` | expected TV with no effect |
| `fork_tv` | the anchor score |
| `tv_debiased` | `fork_tv` − `floor` |
| `signed_dacc` | signed change in accuracy |
| `abs_dacc` | absolute change in accuracy |

- **Ceilings:** 50/50 split-half reliability, Spearman–Brown projected to n = 100. The
  square root is the best correlation any predictor could reach.

## Results

### 1. Target ceilings

| Target | Ceiling, overall | Ceiling, within trajectory |
|---|---|---|
| acc_level | 0.99 | 0.96 |
| entropy | 0.98 | 0.93 |
| floor | 0.98 | 0.95 |
| fork_tv | 0.90 | 0.84 |
| abs_dacc | 0.80 | 0.75 |
| signed_dacc | 0.51 | 0.52 |
| **tv_debiased (the effect itself)** | **0.39** | **0.40** |

With 100 vs 100 resamples, **0 of 6,434** MVP sentences show a significant change in the
answer distribution (permutation test, BH 10%). Phase 2, with 300 vs 300, finds 7 of 190.

### 2. Predictability: best layer per representation

| Target (ceiling) | position | text_S_i | pre | post | delta | pre+post |
|---|---|---|---|---|---|---|
| entropy (0.98) | 0.33 | 0.24 | 0.48 | 0.48 | −0.14 | 0.50 |
| floor (0.98) | 0.33 | 0.23 | 0.52 | 0.53 | −0.14 | 0.55 |
| fork_tv (0.90) | 0.27 | 0.17 | 0.42 | 0.44 | −0.01 | 0.45 |
| abs_dacc (0.80) | 0.23 | 0.14 | 0.27 | 0.28 | 0.02 | 0.29 |
| acc_level (0.99) | 0.21 | −0.08 | 0.16 | 0.16 | −0.20 | 0.17 |
| tv_debiased (0.39) | 0.01 | 0.01 | 0.04 | 0.08 | 0.05 | 0.07 |
| signed_dacc (0.51) | −0.01 | −0.02 | 0.02 | 0.02 | 0.02 | 0.03 |

Within-trajectory Spearman, best layer:

| Target | position | pre | post | pre+post |
|---|---|---|---|---|
| floor | **0.75** | 0.57 | 0.58 | 0.59 |
| entropy | **0.70** | 0.54 | 0.54 | 0.56 |
| fork_tv | **0.57** | 0.42 | 0.42 | 0.43 |
| abs_dacc | **0.43** | 0.30 | 0.31 | 0.31 |
| acc_level | 0.03 | 0.04 | 0.05 | 0.05 |
| tv_debiased | 0.05 | 0.05 | 0.08 | 0.07 |
| signed_dacc | 0.01 | 0.01 | 0.03 | 0.02 |

### 3. What the pre-emission `fork_tv` probe carries (layer 32)

| Comparison | Spearman |
|---|---|
| Prediction vs `fork_tv` | 0.42 |
| Prediction vs `floor` | 0.52 |
| Prediction vs `entropy` | 0.49 |
| Prediction vs `tv_debiased` | 0.05 |
| Prediction vs `acc_level` | −0.01 |
| **Partial correlation with `fork_tv`, controlling for the floor** | **−0.013** |
| Same, also controlling for entropy and position | −0.014 |
| `floor` alone vs `fork_tv` | 0.83 (within trajectory 0.72) |

### 4. Value-probe route

- **The probe.** A probe for accuracy, p̂ from h, reaches only Spearman 0.06–0.16 overall
  and 0.01–0.04 within a trace, even though the target's ceiling is 0.99 / 0.96.
- **The change.** Δp̂ = p̂(h_{i+1}) − p̂(h_i) has Spearman between −0.02 and 0.04 against
  every change target, at every layer.

### 5. Clean labels: the 190 Phase 2 sentences, 300 resamples per fork

- **Labels.** These use 300 resamples per fork, against the MVP's 100.
- **Spearman with the pooled `fork_tv`.** Position gives 0.23, `pre` at layer 8 gives 0.24
  and `text+position` gives 0.26. No representation beats position.
- **AP for the 7 real effects.** Combinations range from 0.04 to 0.24, against a base rate of
  0.037, with no consistent layer or representation. With 7 positives and selection over 50
  combinations, that is not evidence.
- **Effects occur where the model is uncertain.** 5 of the 7 real effects sit in the top
  third of answer entropy (71%, against 33% expected). One-sided Mann–Whitney p = 0.054.

## Root causes, ranked by evidence

1. **The predictable part of the anchor score is answer uncertainty, not anchoring.**
   - `fork_tv` is dominated by its finite-sample floor (Spearman 0.83). The floor is a
     function of how spread out the answers are.
   - The activation probe reads exactly that: Spearman 0.52 with the floor and 0.49 with
     entropy. Its partial correlation with `fork_tv`, once the floor is controlled, is
     **−0.01**.
   - The earlier result (activations 0.23 vs position 0.15, `docs/evaluation_results.md`)
     is therefore an uncertainty probe, not an anchor probe.

2. **The anchor effect itself is barely measurable at 100 resamples.**
   - The effect component has a reliability ceiling of 0.39, and no MVP sentence passes a
     significance test.
   - No representation can be validated against a label that is mostly noise. Every
     representation reaches at most 0.08 on `tv_debiased`, which is 20% of an
     already-low ceiling.

3. **Within a trace, position beats the internal state on everything it can predict.**
   - Uncertainty falls predictably as reasoning proceeds.
   - Within a trajectory, position predicts the floor at 0.75, against 0.57 for
     activations, and `fork_tv` at 0.57, against 0.42.
   - Activations help only *across* problems: entropy 0.48 against 0.33 for position.
     That is problem-level uncertainty.

4. **The stored state does not linearly encode the outcome or its change.**
   - Accuracy is extremely reliable, with a ceiling of 0.96 within a trace, yet it is
     undecodable within a trace (0.04).
   - Δaccuracy is not decodable at all, even from the post-emission state, the change in
     state, or the sentence text.
   - The residual stream at a sentence boundary carries *how uncertain* the model is, not
     *which answer it is heading for* or *whether that answer is right*.
   - So reading the state after the sentence (post, delta) does not rescue prediction.
     The pre-emission design is not the main bottleneck; the readout is.

5. **The data scale limits what can be ruled out.**
   - 20 problems for leave-one-problem-out, one boundary token per sentence, linear
     probes on PCA-128.
   - Nonlinear probes, token-pooled sentence states, or many more problems could still
     find signal.
   - But with the effect label's ceiling at 0.39, any such gain would be hard to
     validate.

## What would be more representative

### Targets

- **Outcome uncertainty (answer entropy, or the floor).**
  - It has a 0.98 ceiling and is cheap to label (it stays reliable at n ≈ 20).
  - Activations decode it across problems (0.48–0.50).
  - It is a principled pre-emission quantity: a sentence can only move the outcome where
    the outcome is still open. The 7 real effects concentrate there (5 of 7 in the top
    entropy tercile).
  - Use it as "anchor opportunity", and as a sampling prior for where to spend
    resamples.
- **Anchors as rare, calibrated events.**
  - Spend 300+ resamples only on high-uncertainty forks, and test with permutation and BH.
  - About a third of forks would need deep sampling, which cuts the cost of collecting
    enough positives roughly threefold.

### Representations (need a GPU)

- **Forced-answer readout.**
  - At each boundary, append the end-of-thinking token and `\boxed{` and read the model's
    answer distribution: P(correct answer string), and the entropy over candidate
    answers.
  - This is a direct readout of the outcome, which the residual probe fails to provide.
  - The change between consecutive boundaries estimates the counterfactual effect with
    one forward pass instead of 100+ resamples.
  - It can be validated against the Phase 2 accuracy curve (ICC 0.98) and the 7 real
    effects.
- **Answer-identity probes, not correctness probes.**
  - Probe for the eventual answer string or class from h, trained on the resample
    outcomes.
  - Correctness depends on ground truth, which the model may not represent. The answer
    it is heading for is a property of its own state.
- **Attention-based importance.** Attention paid to S_i by later tokens (receiver
  heads), and attention-suppression effects, as in the Thought Anchors literature. These
  are post-emission, but they measure how much later reasoning *uses* a sentence. They
  need a forward pass that records attention; none was stored.
- **Sentence-span states.** Pool the residual over all of S_i's tokens, not one boundary
  token; only boundary tokens were stored.

## Not supported (do not pursue)

- More resamples of the continuous `fork_tv` (see `docs/phase2_diagnostics.md`).
- Activation probes on the continuous anchor score as currently defined.
- A value probe (p̂ from h) as an anchor detector.
