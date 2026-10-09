# anchor_reliability_v2: preregistration (plan v1)

**Version 1 — 2026-10-09.** Committed and pushed before any confirmatory data is generated
by this study. Later versions may change PROVISIONAL or OPEN items only before
confirmatory generation, with a dated changelog entry (§12). After confirmatory data
exist, no FIXED item changes.

Labels used: **FIXED** (decided; not changed after confirmatory data), **PROVISIONAL**
(current choice, revisable before confirmatory data with a logged reason), **OPEN**
(undecided; must be resolved before the step that needs it).

## 0. Scope and relation to the MVP

The MVP (repository root at commit `f11823f`: `configs/mvp.yaml`, `src/preanchor/`,
`$PREANCHOR_ROOT` artifacts) is read-only reference material. This study copies the inputs
it needs, with SHA-256 provenance (`inputs/MANIFEST.json`, written by
`scripts/s01_import_inputs.py`), and copies and adapts code with its source recorded. It
never imports the MVP package. All 20 MVP problems have been seen, so every result on them
is **exploratory**. Confirmatory claims about generalisation are reserved for new problems
(Phase 3).

## 1. Audit summary (confirmed facts carried into the design)

These come from the MVP artifacts and the exploratory Phase 1 audit
(`research/redesign/phase1_report.md`), which reproduced every MVP label exactly:

1. With smoothing α = 1e-9, 330 of 344 MVP anchors (96%) draw at least half their KL
   importance from answers that appear among "without" resamples but never among "with"
   resamples. 309 of these are real answers and 21 are `<NO_ANSWER>`. At α = 0.5 the
   share is 23%; at α = 1 it is 14%.
2. Split-half agreement of the MVP anchor flag is κ = 0.19 (N ≈ 50 per half).
3. `<NO_ANSWER>` = 11,181 of 647,400 released samples: 9,739 length-truncated at
   16,384 tokens, 5 parse failures, 1,437 others. About 420 of those others state an
   answer without `\boxed{}`.
4. Two MVP inclusion rules use information from after S_i: the convergence rule looks up
   to 4 forks ahead, and the final-answer rule looks at later sentences.
5. On the MVP label, no baseline or probe is established as better than position.

## 2. Hypotheses

- **H1 (measurement; Phase 2; primary).** The provisional primary score (§3) has
  run-to-run reliability ICC(2,1) ≥ 0.75 at 100 resamples per fork.
- **H2 (binary flag; Phase 2).** Its train-q90 flag has Cohen's κ ≥ 0.60 between
  independent replicates.
- **H3 (prediction; Phase 3, conditional on H1 passing).** On held-out new problems,
  boundary activations add predictive information about the score beyond the strongest
  training-selected observable baseline. This is tested as a nested comparison (§8).

## 3. Estimands and scores

For sentence S_i: **without** = fork-i resamples whose first sentence has MiniLM cosine
< 0.8 to S_i; **with** = fork-(i+1) resamples. Outcome categories come from the v2 parser
(§5).

| Score | Definition | Status |
|---|---|---|
| `tv` | ½ Σ\|p_without − p_with\| | **PROVISIONAL primary** |
| `jsd` | Jensen–Shannon divergence, bits | PROVISIONAL alternative |
| `dacc` | \|acc_without − acc_with\| against ground truth | PROVISIONAL alternative |
| `fork_tv` | TV between all fork-i and all fork-(i+1) samples, no similarity filter | PROVISIONAL alternative |
| `kl_1e-9` | MVP estimand | FIXED as reference only |

- **Primary selection rule (FIXED):** before confirmatory data, the primary score is the
  one with the highest split-half ICC on existing data among {`tv`, `jsd`, `dacc`,
  `fork_tv`}, with ties (within 0.02) broken in that order. The choice is recorded in
  plan v2 before Phase 2 generation.
- **Binary flag (FIXED):** score ≥ its q90 over eligible **training-split** sentences,
  computed on existing released data. That threshold stays fixed for Phase 2.
- **Threshold sensitivity (FIXED, reported):** q80 and q95.
- **Finite-sample floor (FIXED, reported):** the expected score under no true effect,
  i.e. two independent samples of the observed sizes from the pooled distribution.

## 4. Eligibility (FIXED, pre-emission only)

A sentence is eligible when all of the following hold. Each uses only the prefix up to the
boundary, or forks j ≤ i (resampled from prefixes that end before S_i):

1. Its span aligns and the boundary token is valid.
2. No `\boxed{` occurs in sentences before i.
3. Fork i exists.
4. It is not inside a converged run: the forks i−4..i are not all ≤ 0.01 accurate and
   not all ≥ 0.99 accurate.

**Label-definability statuses (not inclusion rules):** `MISSING_FORK` (no fork i+1) and
`NO_SUPPORT` (no dissimilar fork-i resample; affects only the scores that use the filter).
These are reported, not silently dropped.

## 5. Answer parsing and missing answers (FIXED rules; handling PROVISIONAL)

- Parser v2 (`src/arv2/answers.py`, with tests) takes the **last** balanced `\boxed{}`
  after the last `</think>`, or in the whole text if there is none. Every non-answer
  keeps its reason:
  - `length_truncated`: `finish_reason == "length"`, or ≥ 16,320 tokens
  - `unterminated_box`
  - `unboxed_stated`: an explicit "Answer:" statement in the last 400 characters
  - `no_answer`, `empty`
- Normalisation is iterated to a fixed point, and plain numbers are canonicalised.
- **Handling (PROVISIONAL primary):** categories are answers ∪ {`<TRUNCATED>`,
  `<NON_ANSWER>`}, and `unboxed_stated` answers count as answers.
- **Sensitivities (FIXED):**
  1. the v1 parser (MVP categories);
  2. unboxed-stated treated as `<NON_ANSWER>`;
  3. all non-answers dropped, with distributions renormalised.

## 6. Phase 2 design: independent reliability replication

| Item | Value | Status |
|---|---|---|
| Problems | 10 of the 16 non-test MVP problems, by the seeded rule in `research/redesign/phase2_select.py` (seed material `20260914:phase2_selection`), copied to `configs/phase2_forks.json` | FIXED |
| Units | 200 forks (20 contiguous per trajectory) and 190 sentence scores | FIXED |
| Resamples | 2 independent replicates × 100 per fork = 40,000 rollouts | PROVISIONAL (see §9) |
| Generation | Frozen MVP settings: pinned model and vLLM image, temperature 0.6, top_p 0.95, 16,384 new tokens, exact stored prefix, no chat template | FIXED |
| Seeds | per-replicate roots `uint32(sha256("20260914:phase2_replicate:r")[:8])`, then `sha256(root:traj:fork:sample)` | FIXED |
| Interim analysis | none; operational counts only (throughput, failures, truncation) | FIXED |

## 7. Statistics and decision rule (FIXED)

- **Continuous scores:** Spearman ρ and ICC(2,1) (absolute agreement, single
  measurement) between replicate 1 and replicate 2.
- **Binary flags:** Cohen's κ.
- **Uncertainty:** 95% percentile intervals from 2,000 **problem-cluster** bootstrap
  replicates, seeded with `20260914:arv2:bootstrap`.
- **Three-way rule, per statistic:** PASS if the lower bound ≥ threshold; FAIL if the
  upper bound < threshold; INCONCLUSIVE otherwise.
- **Thresholds:** ICC 0.75, Spearman 0.80, κ 0.60.
- **A score form**
  - passes if all its statistics PASS;
  - fails if any statistic FAILS;
  - is inconclusive otherwise.
- **Also reported:**
  - prevalence of the flag in each replicate;
  - within-problem Spearman (median and range over the 10 problems);
  - agreement of each replicate with the released samples (replicate 0);
  - the noise ceiling (§8).
- **If no score passes:** the study reports that the counterfactual-importance target
  cannot be measured reliably at this resample budget, as a result in its own right.

## 8. Prediction evaluation (Phase 3 design; exploratory analyses on MVP data now)

- **Baselines:**
  - position;
  - next-token distribution (entropy and top-k features, plus the probability mass on
    marker tokens such as "Wait", "But", "Alternatively");
  - previous-token identity;
  - bag of the last k tokens;
  - observable combined;
  - activation probes at every stored layer.
- **Protocol:**
  - problem-grouped splits;
  - every scaler, vocabulary, threshold and hyperparameter is fitted on training folds
    only;
  - nested comparisons (baseline vs baseline + activations), tested by the paired
    problem-cluster bootstrap of the AP / Spearman difference.
- **Lead time:** the boundary before S_{i−1} predicting S_i, with rows matched to lead
  zero.
- **Noise ceiling:** the expected correlation between a perfect predictor and a
  single-replicate score ≈ √(reliability). The split-half reliability is projected by
  Spearman–Brown now, and the Phase 2 replicate ICC replaces it later.
- **Phase 3 data (OPEN):** new problems published after the model's training cutoff,
  passing the memorisation check (§10). The primary contrast is the probe against the
  strongest baseline, with the baseline chosen on training data only.

## 9. Sample size and power (procedure FIXED; numbers filled in plan v2)

- Before Phase 2, `scripts/s06_simulate.py` runs a parametric bootstrap. It treats each
  fork's released answer distribution as truth, draws two independent replicates of size
  n ∈ {25, 50, 100, 200, 400}, and computes each score and its reliability over the 190
  planned sentences with problem-clustered intervals. It reports P(PASS), P(FAIL) and
  P(INCONCLUSIVE).
- **Rule:** if at n = 100, P(INCONCLUSIVE) > 0.5 for the primary score, plan v2 raises
  n, or the number of forks, to the smallest simulated value that gives
  P(INCONCLUSIVE) ≤ 0.3, subject to budget. Otherwise n = 100 stands.

## 10. Data validity and contamination (FIXED procedures)

- **Memorisation** (`scripts/s08_memorisation.py`), for all 20 MVP problems:
  1. **No-reasoning accuracy:** the prompt closes the think block immediately; 20 samples
     at temperature 0.6, max 32 tokens.
  2. **Verbatim continuation:** given the first 50% of the problem text, greedy
     continuation of 64 tokens. Report the normalised edit similarity, and exact match of
     the next 10 words.
  - A problem is **flagged** if no-reasoning accuracy ≥ 0.5, or the next 10 words match
    exactly.
  - The task text asks for "14 problems". No 14-problem set is defined in the MVP, so all
    20 are tested (a superset); this discrepancy is recorded.
- **Pathological generation:** for each fork, report the truncation share and the looping
  diagnostics (duplicate-sentence fraction and longest repeated-sentence run in the last
  8,000 characters).

## 11. Stopping rules and budget (FIXED)

- No confirmatory GPU run starts before (a) plan v2 exists with the §3 and §9 outcomes
  filled in, (b) the memorisation report exists, and (c) a measured pilot projection and
  a budget are explicitly approved by the owner.
- Runs stop launching requests when elapsed hours × rate reaches the authorised cost or
  runtime. An early stop is analysed at the largest common N per fork, and that N is
  reported.
- A small throughput pilot (≤ 256 rollouts) is not confirmatory and may run before
  approval.

## 12. Changelog

- v1 (2026-10-09): initial preregistration.
