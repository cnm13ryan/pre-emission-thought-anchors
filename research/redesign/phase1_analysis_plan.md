# Phase 1 analysis rules (written before any reacquired rollout was read)

Written 2026-10-09 at parent commit `f11823f`, before the reacquired per-rollout table
existed. All of Phase 1 is **exploratory** (`stage: exploratory`, `preregistered: false`).
All 20 MVP problems have been seen; no result here is confirmatory. Nothing below
changes `configs/mvp.yaml`.

## R1. Per-answer KL contributions (item 1)

For sentence `i` with "without" counts `n⁻` (dissimilar fork-`i` samples) and "with"
counts `n⁺` (all fork-`i+1` samples) over the union support `K` (including
`<NO_ANSWER>`), the contribution of answer `a` is
`c_a = p⁻_α(a) · log(p⁻_α(a) / p⁺_α(a))`, so `I = Σ_a c_a`. An answer is **absent from
"with"** when `n⁺_a = 0` and `n⁻_a > 0`.

- `absent_share = Σ_{a absent} c_a / I`. An anchor is **driven by absent answers** when
  `absent_share ≥ 0.5`. Fixed in advance; not tuned.
- Secondary: an anchor is **dependent on absent answers** when `I` recomputed with the absent
  categories' `c_a` set to zero falls below that α's anchor threshold.
- Absent contributions are split into `<NO_ANSWER>` and real answers.
- Anchor sets per α: (i) the frozen α = 1e-9 set (threshold 1.355), and (ii) at
  α ∈ {0.5, 1} the set defined by that α's own training-only q90, eligible training rows
  only, computed with the frozen interpolation in `labels/threshold.py`.
- Reproduction gate: the recomputed I at α = 1e-9 must match the stored
  `importance_canonical` (Spearman ≥ 0.999 and identical anchor flags on ≥ 99.5% of rows).
  If the gate fails, the decomposition is reported as `STATUS: REPRODUCTION_FAILED` with
  the mismatch counts.

## R2. `<NO_ANSWER>` classes (item 2)

A sample is `<NO_ANSWER>` when the canonical `normalize_answer` gives `<NO_ANSWER>`.
Classes are assigned in this order:

1. `sample_error`: the sample has an `error` key or an empty rollout.
2. `parse_failure`: the rollout contains `\boxed{` at least once.
3. `length_truncated`: there is no `\boxed{`, and the rollout re-tokenises (pinned
   tokenizer, no special tokens) to at least `16384 − 64` tokens. 16,384 is the upstream
   `max_tokens` default; the 64-token margin allows for differences between re-tokenisation
   and generation.
4. `genuine_non_answer`: there is no `\boxed{`, and the rollout is shorter than that.

The token-length histogram of `<NO_ANSWER>` rollouts is reported. If it has no mass near
the cap, that is stated. The margin is not changed after seeing the histogram.

## R3. Candidate labels (item 3), built without reading any probe output

- (a) `kl_1e-9`: the current `importance_canonical`; anchor `I ≥ 1.355` (frozen q90).
- (b) `kl_kt`: the same KL with α = ½ (Krichevsky–Trofimov / Jeffreys add-½), the
  standard prior for categorical frequencies. The α is chosen by that rule and not by any
  outcome. Anchor = train-only q90 of this label.
- (c) `fork_tv`: a fork-level target, the total-variation distance
  `½ Σ_a |f_i(a) − f_{i+1}(a)|` between the raw empirical answer distributions of **all**
  fork-`i` samples and all fork-`i+1` samples. It uses no similarity filter and no
  smoothing. Binary version = train-only q90.
- (d) `log1p_kl`: continuous `log(1 + I)` at α = 1e-9.

Descriptives per label: support; split prevalence; Spearman against (a); Jaccard of anchor
sets against (a); share of anchors driven by absent answers (R1); and correlation with the
fork-`i` `<NO_ANSWER>` fraction.

Supplementary reliability, using the existing data only: split each fork's samples into
two halves (seeded with `sha256("20260914:split_half:{trajectory}:{fork}")`), compute each
label on each half, and report the Spearman correlation across halves and Cohen's kappa
of the q90 flags (each half's own train q90). Half-samples have N ≈ 50, so these numbers
are a lower bound on full-N reliability and are not a substitute for Phase 2.

## R4. Word-level baselines (item 4)

All features come from tokens at or before `boundary_token_idx`:

- `prev_token`: one-hot identity of the boundary token. Vocabulary = ids seen in ≥ 5
  eligible training rows; all other ids map to OTHER.
- `bow_k`: binary bag of token ids over the last `k` prefix tokens, k ∈ {4, 16, 64}, with
  the same ≥ 5 training-row vocabulary. `k` is selected by train-CV.
- `next_topk`: the top-20 next-token probabilities from `softmax(lm_head(norm(h48)))` in
  float32, as sparse features over the training top-20 vocabulary.
- Reference baselines refit identically: `position` and `observable_combined` (the
  frozen feature lists).

Fitting follows the frozen probe protocol: StandardScaler (`with_mean=False` for sparse
features), L2 logistic with `class_weight="balanced"`, `C` in the frozen grid, and
`StratifiedGroupKFold(4, shuffle)` by problem with the frozen fold seed. Ridge on
continuous labels uses the frozen alpha grid. Selection is by mean train-CV AP (binary)
or Spearman (continuous). Reported: train-CV mean, validation AP and Spearman. **Test-split
metrics are not computed in Phase 1.**

## R5. Inclusion rules available before S_i (item 5)

Each rule is rewritten to use only the stored prefix up to `boundary_token_idx`, plus
fork-`j` samples for `j ≤ i`, which are resampled from prefixes that end before `S_i`.
Removal counts are reported for the original rule and for the rewritten rule, both
marginally (rule alone on all 6,474 rows) and sequentially (in the frozen order).

## Status vocabulary

Every number that cannot be computed is reported with one of these statuses:
`OK`, `REPRODUCTION_FAILED`, `FETCH_FAILED`, `MISSING_FORK`, `NO_SUPPORT`, `NON_FINITE`,
`SINGLE_CLASS`, `NOT_COMPUTED_BY_RULE`, `BLOCKED`.
