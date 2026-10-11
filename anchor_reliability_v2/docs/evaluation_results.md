# Prediction evaluation results (existing data)

**Status:** EXPLORATORY. All 20 MVP problems have been seen; this is not a test of
generalisation (that is Phase 3). Source: `results/s11_evaluation.json`, produced by
`scripts/s11_evaluate.py`.

## Protocol

- **Outer CV:** 5-fold problem-grouped over all 20 problems, so every sentence has one
  out-of-fold prediction.
- **Inner CV:** 4-fold grouped CV selects the hyperparameters, the activation layer from
  {0, 8, 16, 24, 32, 40, 44, 48} and the strongest baseline family.
- **Training-only fitting:** scalers, vocabularies, PCA-256 and the q90 flag threshold.
- **Uncertainty:** 95% problem-cluster bootstrap, 2,000 replicates.

## Targets

- the primary score `fork_tv` (continuous: Spearman);
- its q90 flag (prevalence 0.108: AP);
- the MVP KL anchor (≥ 1.355, prevalence 0.084, 4,112 rows: AP).

## Results

| Model | `fork_tv` Spearman | `fork_tv` flag AP | MVP KL anchor AP |
|---|---|---|---|
| prevalence | — | 0.108 | 0.084 |
| position | 0.152 [0.069, 0.295] | 0.124 [0.070, 0.209] | 0.097 [0.044, 0.164] |
| observables | 0.141 [0.061, 0.263] | 0.135 [0.083, 0.213] | 0.105 [0.048, 0.181] |
| markers (next-token marker mass) | 0.022 [−0.030, 0.089] | 0.123 [0.080, 0.181] | 0.088 [0.037, 0.156] |
| next_top20 | 0.047 [0.000, 0.105] | 0.132 [0.075, 0.196] | 0.110 [0.049, 0.183] |
| prev_token | −0.061 [−0.146, 0.043] | 0.127 [0.071, 0.202] | 0.088 [0.038, 0.169] |
| bow16 | 0.121 [0.068, 0.191] | 0.138 [0.067, 0.202] | 0.086 [0.032, 0.170] |
| strongest baseline (inner-CV choice) | 0.152 [0.069, 0.295] | 0.120 [0.081, 0.187] | 0.100 [0.044, 0.180] |
| activation | 0.229 [0.154, 0.303] | 0.151 [0.081, 0.241] | 0.171 [0.046, 0.315] |
| observables + activation | 0.231 [0.164, 0.316] | 0.173 [0.093, 0.280] | 0.125 [0.046, 0.240] |
| strongest + activation (fold's selected baseline + activation) | 0.196 [0.134, 0.282] | 0.157 [0.087, 0.262] | 0.114 [0.051, 0.206] |

The strongest baseline was position in all 5 outer folds for the continuous score, and
mixed (observables / position / next_top20) for the flag.

## Nested comparisons (paired problem-cluster bootstrap of the difference)

| Comparison | `fork_tv` | flag AP | MVP KL anchor AP |
|---|---|---|---|
| activation − position | +0.077 [−0.069, +0.182] | +0.027 [−0.011, +0.057] | +0.074 [−0.044, +0.162] |
| activation − strongest | +0.077 [−0.069, +0.182] | +0.031 [−0.009, +0.062] | +0.072 [−0.012, +0.146] |
| (obs + act) − obs | +0.090 [−0.003, +0.166] | +0.039 [−0.012, +0.082] | +0.020 [−0.021, +0.068] |
| (strongest + act) − strongest | +0.045 [−0.043, +0.107] | +0.036 [−0.002, +0.085] | +0.014 [−0.009, +0.041] |

- No interval of the 12 excludes zero.
- **Correction (2026-10-11, Codex review on PR #5).** The earlier "strongest + activation"
  model searched only three fixed baseline families, so it was not nested in the
  strongest-baseline comparison. In fold 4 of the flag target, for example, the strongest
  baseline was `next_top20`, but the augmented model could not use it. Each fold now
  augments exactly the baseline family that fold selected, and inner CV chooses only the
  layer and penalty.
  - The corrected flag comparison is +0.036 [−0.002, +0.085], down from +0.048
    [+0.001, +0.093]. This was the one interval that previously excluded zero.
  - The other two corrected comparisons also shrink: +0.045 for `fork_tv` and +0.014 for
    the KL anchor.
  - No other model or comparison changed.
- The 12 comparisons are not corrected for multiplicity, and the data are exploratory.
  **This is not evidence of incremental information.**
- The point estimates are consistently positive. That direction is what Phase 3 tests
  confirmatorily.

## Layer curve, lead time and noise ceiling

**Layer curve** (activation-only AP on the flag, inner-selected C):

| Layer | 0 | 8 | 16 | 24 | 32 | 40 | 44 | 48 |
|---|---|---|---|---|---|---|---|---|
| AP | 0.130 | 0.127 | 0.130 | 0.137 | 0.161 | 0.161 | 0.151 | 0.157 |

- Inner CV chose layers 24–44, mostly 32 and 40.
- Even layer 0 (embeddings) reaches 0.130, against prevalence 0.108. Much of the signal
  is available without deep processing, consistent with position and lexical effects.

**Lead time** (matched rows, 4,699 sentences with a previous boundary):

| Target | Lead 0 | Lead 1 | Lead 0 − lead 1 |
|---|---|---|---|
| `fork_tv` Spearman | 0.202 | 0.197 | +0.005 [−0.036, +0.047] |
| Flag AP | 0.165 | 0.140 | +0.025 [−0.012, +0.072] |

There is no detectable advantage of the immediate boundary over the boundary one sentence
earlier.

**Noise ceiling:** projected full-N reliability of `fork_tv` is 0.76, so the maximum
expected correlation of any predictor with a single-replicate score is ≈ 0.87. The best
observed Spearman (0.23) is far below the ceiling, so measurement noise is not the only
limit on prediction.

## Reading

- **Confirmed:** under strict grouping and training-only fitting, activations score above
  every lexical and next-token baseline in point estimate, on all three targets.
- **Not established:** that this advantage holds beyond chance. The intervals include 0
  in 11 of 12 comparisons.
- **Interpretation (tentative):** most of the predictable variance is shared with
  position and context. Earlier layers already carry part of it, and the earlier boundary
  works as well as the immediate one. That favours slowly varying trajectory state over
  sentence-specific pre-emission planning.
