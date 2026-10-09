# Revised evaluation design (Phase 2 → Phase 3)

Status labels: **FIXED**, **PROVISIONAL** or **OPEN**, as in `phase2_plan.md`.

## A. What Phase 2 must establish first (FIXED)

Phase 3 (prediction on new problems) runs only for score forms that **PASS** the Phase 2
three-way rule (§7 of the plan).

- If the primary score is INCONCLUSIVE, Phase 3 is not run on it. The owner decides
  between more resamples (sized by `s06_simulate.py`) and stopping.
- If every form FAILS, the study reports that the target is not measurable at this
  budget, and Phase 3 is not run.

## B. Phase 3: held-out problem generalisation

| Item | Choice | Status |
|---|---|---|
| Problem source | Competition problems with numeric answers published after the base model's training data (e.g. AIME 2025 I/II, HMMT February 2025). The exact list is fixed before any trace is generated. | OPEN (list) / FIXED (rule) |
| Contamination controls | (1) publication date after the cutoff; (2) the `s08_memorisation.py` protocol, where flagged problems are excluded before generation; (3) a duplicate check against the 20 MVP problems and MATH, by exact and normalised text match. | FIXED |
| Traces | One base trajectory per problem from the pinned model at the MVP settings. Sentences are segmented with the upstream rule. A correct/incorrect pair is not required. | PROVISIONAL |
| Labels | The Phase 2 passing score with its Phase 2 resample budget. Thresholds are fixed from the MVP training split and not refit on Phase 3. | FIXED |
| Sample size | Precision-based: interval half-width is assumed to scale as 1/√(number of problems), starting from the MVP test interval at 4 problems. If 0.132 is the half-width, a target half-width of 0.05 needs ≈ 4·(0.132/0.05)² ≈ 28 problems. If 0.132 is the full width (half-width 0.066), it needs ≈ 7. Which reading applies must be confirmed. | OPEN |
| Primary contrast | (strongest baseline + activations) − (strongest baseline), with the baseline family, layer and hyperparameters chosen by grouped CV on the 20 MVP problems only, and frozen before Phase 3 labels exist. Metric: AP of the q90 flag (binary) and Spearman (continuous). Paired problem-cluster bootstrap, 2,000 replicates. | FIXED |
| Decision | Report the interval; claim incremental information only if the lower bound > 0. | FIXED |
| Secondary | Lead-1 vs lead-0 on matched rows; layer curve; per-problem paired differences; noise ceiling from the Phase 2 ICC. | FIXED |

## C. Baselines (implemented in `s11_evaluate.py`)

All are pre-emission. Within each problem-grouped training fold they get training-only
scaling, vocabularies, PCA and flag thresholds.

- **position:** `sentence_idx`, generated tokens and context fraction.
- **observables:** the 12 MVP observables.
- **markers:** next-token probability mass on reasoning-marker tokens (backtrack, plan,
  conclude, check groups), plus entropy.
- **next_top20:** sparse top-20 next-token distribution.
- **prev_token:** identity of the previous token.
- **bow16:** bag of the last 16 token ids.
- **strongest_baseline:** the family chosen by inner CV from the above.
- **activation:** PCA-256 of one residual layer from {0, 8, 16, 24, 32, 40, 44, 48},
  with the layer chosen by inner CV. A per-layer curve is also reported.
- **Nested:** observables + activation, and strongest + activation.

## D. Representative trace examples (FIXED procedure)

For the passing score, deciles are computed on eligible sentences. Two sentences are drawn
per decile with seed material `20260914:arv2:examples`. Each example shows its prefix
tail, its "without" and "with" answer distributions, its score and its floor. Examples are
illustrative only and are not selected by outcome.

## E. Unresolved

- The 14-problem set named in the task is undefined. The memorisation check covers all 20.
- Which reading of the 0.132 MVP test width applies (half-width or full width).
- Whether Phase 3 needs correct/incorrect trajectory pairs.
