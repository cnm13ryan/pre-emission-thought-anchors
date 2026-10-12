# Data validity: parsing, missing answers, truncation, looping, eligibility

**Status:** EXPLORATORY; executed. Sources: `results/s04_summary.json` and
`results/s07_validity.json`. Tests: `tests/test_core.py` and `tests/test_eligibility.py`.

## 1. Final-answer parsing (v1 → v2)

Out of 647,400 released rollouts:

| v2 status | Rollouts |
|---|---:|
| answered (last balanced box) | 635,882 |
| length_truncated (≥ 16,320 tokens, no complete box) | 10,027 |
| no_answer | 1,133 |
| unboxed_stated (prose "Answer:") | 355 |
| unterminated_box | 3 |

- **v1 defects confirmed:**
  1. **First box instead of last.** For example, 136 rollouts change from `-\frac12` to
     `\frac15`, and 25 from `1280` to `640`; in each case the first box was an
     intermediate value.
  2. **Unterminated boxes accepted as answers.** v2 counts 10,027 truncations against
     v1's 9,739 truncated `<NO_ANSWER>`. 336 rollouts have a v1 answer but no v2 answer.
  3. **Format variants split.** `5.0` and `5` were separate categories (1,442 rollouts),
     as were `6.30` and `6.3`.
  4. **Thousands separators read as decimals.** `normalize_latex("1,000")` gives `1.000`.
     This is found and fixed in v2, with a test.
- **Scale:** 2,246 answered rollouts (0.35%) change category. v2 correctness agrees with
  the stored upstream correctness on 99.95% of answered rollouts.
- **No spurious merges:** across 453,150 single-box rollouts, the only v2 value absorbing
  two v1 values is `5` / `5.0`, which is intended.

## 2. Missing-answer and truncation handling (plan §5)

- **Primary categories:** answers ∪ {`<TRUNCATED>`, `<NON_ANSWER>`}, with unboxed-stated
  answers counted as answers.
- **Sensitivities:** Spearman of `tv` against `tv` computed with each alternative
  handling:

| Handling | Spearman vs primary `tv` |
|---|---:|
| v1 parse | 0.995 |
| unboxed counted as `<NON_ANSWER>` | 0.999 |
| non-answers dropped | 0.974 |

The bounded score is therefore insensitive to these handling choices. The MVP KL is not:
its anchor flags change on 46 rows under v2 parsing alone, and its anchor count goes from
344 to 366.

## 3. Pathological looping and truncation

- **Loop signature:** duplicate-sentence fraction ≥ 0.5, or a run of ≥ 5 identical
  sentences, in the last 8,000 characters.
  - 10% of truncated rollouts show it.
  - 0.06% of non-truncated rollouts show it.
  - Most truncations are therefore long, non-repeating reasoning, not loops.
- **Forks with high truncation:** 260 forks have ≥ 10% truncated samples; 2 forks have
  ≥ 50%.
- **Concentration:** problem 3550 (`3550:correct` 14.7% truncated, `3550:incorrect` 8.8%),
  then `4164:incorrect` (5.5%). The per-fork table is `results/s07_fork_pathology.csv`.

## 4. Eligibility without future information

| Rule | MVP rule removes | Pre-emission v2 rule removes |
|---|---:|---:|
| convergence | 1,835 (looks up to 4 forks ahead) | 1,723 (forks i−4..i) |
| final answer | 19 (uses later sentences) | 1 (box in an earlier sentence) |
| next fork / dissimilar support | inclusion rules | label statuses: MISSING_FORK 11, NO_SUPPORT 627 |

- v2 eligibility is 4,750 sentences. All 4,019 MVP-eligible sentences are among them, and
  731 are new.
- Leakage tests mutate the future and assert that v2 eligibility is unchanged. They also
  show that the MVP rules do change.

## 5. Recomputed MVP statistics

- **Reproduction:** the KL label at α = 1e-9 with v1 categories reproduces all 4,019 MVP
  values exactly (maximum difference 2.7e-15).
- **On v2-eligible rows:**

| Statistic | v1 parse | v2 parse |
|---|---:|---:|
| KL anchors (≥ 1.355) | 344 | 366 |
| Prevalence, train / validation / test | 0.098 / 0.078 / 0.035 | 0.102 / 0.089 / 0.041 |

- **Prediction results under v2:** `docs/evaluation_results.md`.
