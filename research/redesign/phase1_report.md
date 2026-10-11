# Redesign Phase 1 report: what the anchor label measures

**Status: EXPLORATORY.** `stage: exploratory`, `preregistered: false`. Every number here
comes from the 20 MVP problems, all of which have been seen before. No result is
confirmatory. CPU/MPS only; nothing was generated. Parent commit `f11823f`;
`configs/mvp.yaml` is unchanged. Adopting any label, rule or baseline below would need a
written amendment, a new config fingerprint and run ID, and owner approval (AGENTS §4).

The rules were written in [`phase1_analysis_plan.md`](phase1_analysis_plan.md) before the
reacquired rollouts were read. No probe output was read. No test-split metric was computed.

## Status summary

| Item | Status | Headline |
|---|---|---|
| 1. Per-answer KL decomposition | OK | At α = 1e-9, 330 of 344 anchors (95.9%) get at least half their I from answers absent from the "with" group. 309 of these are real answers and 21 are `<NO_ANSWER>`. At α = 0.5 the share is 23.0%; at α = 1 it is 14.4%. |
| 2. `<NO_ANSWER>` split | OK, with a caveat | 11,181 of 647,400 samples: 9,739 length-truncated, 5 parse failures, 1,437 "genuine non-answer", 0 sample errors. About 29% of the "genuine" class state an unboxed answer (post hoc, descriptive). |
| 3. Candidate labels | OK | Four labels are built on the 4,019 frozen-eligible rows. Split-half reliability is low for every label and lowest for the current one. |
| 4. Word-level baselines | OK (train-CV and validation only) | On the current label, the top-20 next-token distribution has the best train-CV AP (0.140), against 0.114 for position. On validation no baseline clearly beats position (0.133). |
| 5. Pre-S_i inclusion rules | OK | The rewritten rules keep 4,750 rows instead of 4,019. The similarity-support and next-fork rules cannot be rewritten; they become label statuses. |
| 6. Design-doc edit | **BLOCKED** | No pipeline-figure placeholder exists in any `.md` in the repo. The owner said the doc is elsewhere but gave no location. |
| Test-split metrics | NOT_COMPUTED_BY_RULE | Not loaded in Phase 1. |

## Data and reproduction

- **Reacquisition.** All 6,474 released fork files (21.0 GB) were fetched at dataset
  revision `a3a2fe1e…` into memory. Each was checked against the size, Git blob SHA-1 and
  SHA-256 recorded in the Stage 4 raw run's `source_manifest.json`, then reduced to a
  65 MB per-rollout table at
  `~/preanchor-root/data/processed/redesign_phase1/rollouts/`. Raw payloads were not kept.
  The first pass hit 394 `HTTP 429` (rate-limit) failures. Two resumable retry passes
  cleared them: 8 failures remained after the second pass and 0 after the third.
  **Final FETCH_FAILED: 0.**
- **Similarity.** Pinned MiniLM (`1110a243…`) ran on MPS: 647,400 pairs, 0 NaN.
- **Reproduction gate (R1): PASSED, exactly.**

| Check | Result |
|---|---|
| Eligible rows recomputed | 4,019 / 4,019 (status OK) |
| `dissimilar_count` identical to Stage 4 | 100% |
| Spearman, recomputed vs stored I (α = 1e-9) | 1.0000 |
| Max abs diff in I at α = 1e-9 / 0.5 / 1 | 1.4e-17 / 1.4e-17 / 1.1e-16 |
| Anchor flags identical at 1.355 | 100% |

- **Word features.** Re-tokenisation reproduces Stage 7 `token_ids_sha256` for all 40
  trajectories (0 mismatches). Next-token distributions are recomputed in bf16 on MPS from
  the stored layer-48 states with the pinned final norm and `lm_head`. Error against the
  stored entropy: median 0.011 nats, p99 0.059, max 0.091, which is expected bf16
  MPS-vs-CUDA kernel drift.
- **Environment.** An untracked analysis venv with the versions pinned in `uv.lock`;
  recorded in `docs/environment_deviations.md`. `uv.lock` and `.venv` are untouched.

## 1. Which answers drive each anchor

For each sentence, `I = Σ_a c_a` with `c_a = p⁻(a) log(p⁻(a)/p⁺(a))`. An answer is
*absent* when `n⁺_a = 0 < n⁻_a`. An anchor is *driven* when absent answers supply at least
half of I. It is *dependent* when removing their contributions drops I below that α's
threshold. Each α's threshold is its own training-only q90.

**All 4,019 eligible rows** (training-only rows give the same picture: 95.8%, 26.8% and
15.5% driven at the three α values):

| α | Threshold (train q90) | Anchors | Driven by absent | of which `<NO_ANSWER>` | of which real answer | Dependent on absent | Non-anchors driven |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1e-9 | 1.355 | 344 | **330 (95.9%)** | 21 | 309 | 329 (95.6%) | 36.0% |
| 0.5 | 0.2462 | 400 | 92 (23.0%) | 14 | 78 | 104 (26.0%) | 36.2% |
| 1 | 0.2656 | 404 | 58 (14.4%) | 8 | 50 | 68 (16.8%) | 37.8% |

**The frozen α = 1e-9 anchor set (344), re-examined at each α:**

| α | Still driven by absent | Still an anchor at that α's own threshold |
|---|---:|---:|
| 0.5 | 316 (91.9%) | 117 (34.0%) |
| 1 | 314 (91.3%) | 86 (25.0%) |

What this shows (exploratory):

- The review's diagnosis holds: under α = 1e-9 the anchor label is almost entirely the
  detection of an answer that appears among "without" reruns and never among "with"
  reruns. The ~22-nat log ratio dominates I as soon as such an answer reaches a few percent.
- It holds with one correction. The absent answers are **mostly real answers** (309 of
  330), not `<NO_ANSWER>` (21 of 330). `<NO_ANSWER>` appears in the "without" group of 70%
  of anchors, but it usually appears on both sides, so it rarely drives KL.
- About 36% of non-anchors are also mostly absent-answer KL, but with too little mass to
  cross 1.355. So the threshold is effectively a cut on the size of the rarest unseen answer.

## 2. What `<NO_ANSWER>` is made of

Classes were assigned by rule R2, fixed in advance: `sample_error`, then `parse_failure`
(any `\boxed{`), then `length_truncated` (no box, ≥ 16,320 re-tokenised tokens), then
`genuine_non_answer`.

| Class | Samples | Forks with ≥ 1 | Mean per fork | p95 per fork | Max per fork | In eligible "without" groups (rows / samples) | In anchors' "without" groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| sample_error | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 |
| parse_failure | 5 | 5 | 0.001 | 0 | 1 | 4 / 4 | 2 |
| length_truncated | 9,739 | 1,620 | 1.50 | 7 | 50 | 917 / 5,867 | 212 |
| genuine_non_answer | 1,437 | 812 | 0.22 | 1 | 42 | 388 / 668 | 62 |
| **All `<NO_ANSWER>`** | **11,181** (1.73% of samples) | **2,124 of 6,474** | 1.73 | 8 | 50 | — | — |

Forks by `<NO_ANSWER>` count: 4,350 have 0; 1,008 have 1; 650 have 2–5; 207 have 6–10;
123 have 11–25; 136 have 26–50; 0 have more than 50.

**Token-length check.** Re-tokenised lengths of `<NO_ANSWER>` rollouts: 425 under 2k;
345 at 2–4k; 465 at 4–8k; 140 at 8–12k; 61 at 12–16k; 1 at 16,000–16,319; **9,740 at
16,320–16,384**; 4 above 16,384. There is a sharp spike at the 16,384-token cap, so the
truncation class is well identified. Only 1.8% of truncated rollouts contain `</think>`.

**Caveat: `genuine_non_answer` is mixed.** This split is post hoc and descriptive; the
pre-stated rule is unchanged. Among the 1,437 rollouts, 420 (29%) end by stating an answer
in prose without `\boxed{}`, for example `**Answer:** 6.4 cm`. These are format failures
that a looser parser would score. The other 1,017 stop mid-reasoning well below the cap
(median 3,860 tokens). Possible causes are a provider-side stop or an early EOS; that
cannot be told apart from the saved text. 35% contain `</think>`. They concentrate in
problem 6596 (483) and 4164 (217).

Concentration: problem 3550 accounts for 6,770 of 9,739 truncations (3,721 in
`3550:correct`, 3,049 in `3550:incorrect`). 14.2% of all samples in `3550:correct` are
`<NO_ANSWER>`. The full per-fork table is
[`tables/no_answer_by_fork.csv`](tables/no_answer_by_fork.csv) (6,474 rows) and the
per-trajectory table is
[`tables/no_answer_by_trajectory.csv`](tables/no_answer_by_trajectory.csv).

## 3. Candidate labels, built without reading probe output

| Label | Definition | Rows | Missing | Train-q90 threshold | Prevalence train / val / test | Spearman vs (a) | Jaccard of anchor set vs (a) | Spearman vs "without" `<NO_ANSWER>` fraction | Anchors with any `<NO_ANSWER>` in "without" |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| (a) `kl_1e-9` | current `importance_canonical` | 4,019 | 0 | 1.355 | 0.100 / 0.079 / 0.036 | 1 | 1 | 0.452 | 69.8% |
| (b) `kl_kt` | same KL, α = ½ (KT/Jeffreys rule) | 4,019 | 0 | 0.2462 | 0.100 / 0.104 / 0.093 | 0.705 | 0.187 | 0.244 | 24.5% |
| (c) `fork_tv` | TV distance between all fork-i and all fork-(i+1) answers; no filter, no smoothing | 4,019 | 0 | 0.170 | 0.102 / 0.081 / 0.046 | 0.575 | 0.211 | 0.343 | 58.4% |
| (d) `log1p_kl` | log(1 + I), α = 1e-9 | 4,019 | 0 | — (continuous; ranks identical to a) | — | 1 | — | 0.452 | — |

Test prevalences are counts of label values only. No model was scored on test.

Disclosure: α = 0.5 was already a frozen sensitivity, and its overall Spearman with (a)
(0.705) had been reported in `research/blog/evidence.json`. Rule (b) was therefore chosen
on a principle (the KT add-½ prior), not blind to every property of the resulting label.

**Supplementary split-half reliability.** Each fork's samples were split in half with a
seed; each label was computed per half (N ≈ 50 per half; about 25 dissimilar "without"
samples). This is a lower bound on full-N reliability, not a substitute for Phase 2.

| Label | Rows | Dropped (NO_SUPPORT in a half) | Spearman, half A vs B | Pearson on log1p | Spearman–Brown projection to full N (rough) | Cohen's κ, q90 flags |
|---|---:|---:|---:|---:|---:|---:|
| `kl_1e-9` | 3,902 | 117 | 0.444 | 0.400 | ≈ 0.57 | **0.195** |
| `kl_kt` | 3,902 | 117 | 0.555 | 0.577 | ≈ 0.73 | 0.445 |
| `fork_tv` | 4,019 | 0 | 0.533 | 0.619 | ≈ 0.76 | 0.417 |

Exploratory reading: the current binary anchor flag barely agrees with itself across
independent halves of the same reruns (κ ≈ 0.19). Both alternatives roughly double κ, but
none reaches a level usually called reliable. This is evidence for the review's diagnosis;
it does not settle which label to adopt. Phase 2's independent reruns are the actual test.

## 4. Word-level baselines

These follow the frozen probe protocol: the same seed and `StratifiedGroupKFold(4)` by
problem, the same C/alpha grids and balanced L2 logistic regression. Vocabularies are built
inside each training fold. Selection uses train-CV only.

**Protocol check:** `position` and `observable_combined` on label (a) reproduce the frozen
validation APs exactly (0.1330 and 0.1266, as in `research/blog/evidence.json`).

`bow_k` selects k from {4, 16, 64} by CV. `observables+word` = all 12 observables plus
`prev_token`, `bow_k` and `next_top20`.

**Binary labels: AP** (train-CV mean [fold min–max] / validation)

| Baseline | (a) `kl_1e-9` | (b) `kl_kt` | (c) `fork_tv` |
|---|---|---|---|
| prevalence | 0.089 / 0.079 | 0.100 / 0.104 | 0.094 / 0.081 |
| position | 0.114 [0.041–0.172] / **0.133** | 0.123 [0.092–0.147] / 0.140 | **0.177** [0.048–0.366] / 0.077 |
| observable_combined | 0.099 [0.029–0.178] / 0.127 | **0.173** [0.122–0.298] / **0.147** | 0.149 [0.053–0.250] / 0.077 |
| prev_token | 0.096 [0.038–0.198] / 0.104 | 0.114 [0.084–0.132] / 0.102 | 0.095 [0.050–0.158] / 0.081 |
| bow_k (selected k) | 0.107 (k = 64) / 0.094 | 0.120 (k = 16) / 0.127 | 0.112 (k = 64) / 0.140 |
| next_top20 | **0.140** [0.057–0.251] / 0.130 | 0.149 [0.113–0.180] / 0.122 | 0.113 [0.042–0.176] / 0.132 |
| observables+word | 0.134 [0.058–0.259] / 0.069 | 0.162 [0.122–0.183] / 0.122 | 0.133 [0.046–0.221] / **0.144** |

Validation positives: 54 for (a), 71 for (b), 55 for (c), out of 680 rows. Validation
AUROCs are in [`tables/item4_word_baselines.csv`](tables/item4_word_baselines.csv).

**Continuous labels: Spearman** (train-CV mean / validation)

| Baseline | (d) log1p KL | log1p (b) | (c) TV |
|---|---|---|---|
| position | **0.253** / **0.282** | **0.196** / **0.234** | **0.299** / **0.340** |
| observable_combined | 0.083 / −0.162 | 0.142 / 0.100 | 0.267 / 0.333 |
| prev_token | 0.025 / 0.097 | −0.010 / 0.114 | 0.063 / 0.147 |
| bow_k | 0.113 / 0.017 | 0.056 / 0.026 | 0.144 / 0.235 |
| next_top20 | 0.120 / 0.074 | 0.114 / 0.027 | 0.089 / 0.077 |
| observables+word | 0.105 / 0.150 | 0.119 / 0.073 | 0.180 / 0.172 |

What this shows (exploratory):

- Under the train-CV rule, the strongest baseline for the current label is
  `next_top20`. The full next-token distribution carries anchor-relevant information that
  the scalar uncertainty features discard. On validation it is not distinguishable from
  position (0.130 vs 0.133).
- For every label, fold-to-fold spread (e.g. 0.048–0.366) is larger than the differences
  between baselines. With 12 training problems, selecting the "strongest baseline" is
  itself noisy. Phase 3 has to allow for that.
- Position dominates the continuous targets.
- The combined `observables+word` model overfits label (a): train-CV 0.134, validation
  0.069.

## 5. Inclusion rules using only information available before S_i

"Before S_i" means the stored prefix up to `boundary_token_idx`, plus fork-j samples for
j ≤ i, which are resampled from prefixes that end before S_i. Counts are over all 6,474
sentence rows.

| Frozen rule | Information it uses beyond the boundary | Rewritten rule | Removed, original: marginal / sequential | Removed, rewritten: marginal / sequential |
|---|---|---|---|---|
| exact span alignment | S_i's text span | unchanged; all spans align | 0 / 0 | 0 / 0 |
| not first token | none | unchanged | 0 / 0 | 0 / 0 |
| not final-answer sentence (i ≥ last `\boxed` sentence) | S_i's text and **later** sentences | drop i if a `\boxed{` occurs in a sentence before i | 19 / 19 | 1 / 1 |
| next fork available | fork i+1, which conditions on S_i | require fork i only; a missing fork i+1 becomes label status MISSING_FORK | 40 / 21 | 0 / 0 (11 rows get MISSING_FORK) |
| before convergence (first j where forks j..j+4 are all extreme; drop i ≥ j) | forks j+1..j+4, up to 4 forks **after** i | drop i if forks i−4..i are all ≤ 0.01 or all ≥ 0.99 accurate (drops i ≥ j+4) | 1,835 / 1,807 | 1,723 / 1,723 |
| ≥ 1 dissimilar replacement | similarity of replacements to S_i's text | **cannot be rewritten**: the KL label is conditioned on S_i by design; it becomes label status NO_SUPPORT | 874 / 608 | — (627 rows NO_SUPPORT) |
| finite, nonnegative I | the label | label status NON_FINITE (0 rows) | 0 / 0 | — |
| **Remaining** | | | **4,019** | **4,750** |

Of the 4,750 rows kept by the rewritten rules: 4,112 have a defined KL label; 627 are
NO_SUPPORT for KL; 11 are MISSING_FORK. `fork_tv` is defined on 4,739, because it needs no
similarity filter. All 4,019 frozen-eligible rows are inside the 4,750. Of the 731 new
rows, none with a defined KL label is an anchor at 1.355; most are converged-region and
NO_SUPPORT rows.

Two rules use information from after S_i: the convergence rule (up to four later forks)
and the final-answer rule (later sentences). The similarity-support rule cannot be made
pre-S_i at all while the label depends on S_i's dissimilar replacements. The fork-level
target (c) avoids that dependence.

## 6. Design document: BLOCKED

None of the tracked or untracked `.md` files in the repository has a missing image link or
a pipeline-figure placeholder. The owner said the design doc is not `research/blog/post.md`
or `research/figure_plan/figure_plan.md` and is located elsewhere, but gave no location.
Status: **BLOCKED until the location is provided.** For reference, the cut-off in
repository docs is written variously as 1.3549810893, 1.354981, 1.35498 and 1.355. The edit
to "1.355 everywhere" applies only to the design doc; frozen artifacts keep the exact value
`1.3549810893213545`.

## Limitations

- Exploratory on 20 seen problems: 12 training, 4 validation; test not used.
- Split-half reliability uses about 25 dissimilar samples per half and understates
  full-N reliability. The Spearman–Brown projections are rough.
- The `<NO_ANSWER>` classes rest on re-tokenised lengths and the upstream default
  `max_tokens = 16384`. The provider's actual cap is not recorded in the released data.
- Similarity was recomputed on MPS rather than CUDA. It reproduced every
  `dissimilar_count` exactly, so this had no effect here.
- Selecting baselines with 4 folds of 3 problems each is high-variance (see fold ranges).

## Artifacts and provenance

| Artifact | SHA-256 |
|---|---|
| Input `stage4/labels.parquet` | `fab9886e478922ac132d0a98f3320c3047cb87cd6252c0d06117b99861a61b31` |
| Input `stage7/features.parquet` | `f3227c386383701741aa284cc3bbf6b7c3205f2de9593616575bf1187e9a064d` |
| `redesign_phase1/candidate_labels.parquet` | `d002669da24a0a0b77430b38b9d0953b83f6c8c06b60eac3f8a4fa4a17775d07` |
| `redesign_phase1/similarity.parquet` | `2d764993d0071788f790ec1ad43eb3acdf13c5c8ec22dffa27fb50f010585399` |
| `redesign_phase1/word_features.parquet` | `8250b5932819df73a1f209732e4c4b5f7cf60692f668c24b71fa705e05b563b4` |
| `phase1_analysis_plan.md` (rules) | `414e9cb8a63819e0d2b79f65425849eaf04d53e55516415d813fec1acb308139` |

Scripts are in `research/redesign/`: `fetch_rollouts.py`, `similarity.py`,
`word_features.py`, `phase1_labels.py`, `phase1_baselines.py`. Tables are in
`research/redesign/tables/`. Large data is under
`~/preanchor-root/data/processed/redesign_phase1/`. Model and tokenizer caches are under
`~/preanchor-root/cache/hf` (not backed up).

Regenerate with:

```
PY=~/preanchor-root/cache/redesign-venv/bin/python
PYTHONPATH=src $PY research/redesign/fetch_rollouts.py --out ~/preanchor-root/data/processed/redesign_phase1/rollouts
$PY research/redesign/similarity.py --root ~/preanchor-root/data/processed/redesign_phase1
$PY research/redesign/word_features.py --out ~/preanchor-root/data/processed/redesign_phase1/word_features.parquet
$PY research/redesign/phase1_labels.py
$PY research/redesign/phase1_baselines.py
```

Phase 1 stops here. Phase 2 (GPU) has not been planned or started.
