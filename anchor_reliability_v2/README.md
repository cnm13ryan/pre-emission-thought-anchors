# anchor_reliability_v2

A standalone follow-up to the pre-emission thought-anchor MVP. It asks:
1. Can the counterfactual-importance target be measured reliably?
2. Once the measurement is repaired, do pre-emission activations predict it beyond strong
   baselines?

- **Isolation:** the MVP (the rest of this repository at `f11823f`) is read-only input.
  This folder never imports `preanchor`. Copied code names its source in its docstring
  (the upstream answer parser keeps its MIT notice). Copied data carries SHA-256
  provenance in `$ARV2_DATA/inputs/MANIFEST.json`, and the 3.1 GB of activation shards
  are checksummed and read in place.
- **Branch:** `study/anchor-reliability-v2`.
- **Status of results:** every result on the 20 MVP problems is **exploratory**.

## Layout

| Path | Contents |
|---|---|
| `phase2_plan.md` | Preregistration, v1 → v2.1 with a changelog |
| `src/arv2/` | `answers` (v1 verbatim + v2 parser), `scores` (KL/TV/JSD/Δacc), `reliability` (ICC, κ, clustered bootstrap, three-way rule), `eligibility` (pre-emission rules), `vllm_client`, `paths` |
| `scripts/s01…s11` | Pipeline (see Reproduce) |
| `tests/` | 32 tests: maths, parsing, leakage, baseline correctness, integrity, reproducibility |
| `docs/` | `reliability_diagnostics`, `data_validity`, `memorisation_report`, `evaluation_design` (Phase 2/3), `evaluation_results`, `compute_budget` |
| `results/` | Small JSON/CSV outputs |
| `configs/phase2_forks.json` | Fixed Phase 2 selection |

## Reproduce

Environment: Python 3.11 with `requirements.txt`. The versions match the MVP `uv.lock`.

```
export ARV2_DATA=~/preanchor-root/anchor_reliability_v2   # study data root (outside git)
export ARV2_MVP_ROOT=~/preanchor-root                      # read-only MVP artifacts
python scripts/s01_import_inputs.py         # copy inputs + SHA-256 manifest
python scripts/s02_fetch_rollouts.py        # reacquire 6,474 forks (21 GB streamed, hash-checked)
python scripts/s03_similarity.py            # MiniLM replacement similarity
python scripts/s04_scores.py                # scores, eligibility, split halves; MVP reproduction gate
python scripts/s05_reliability.py           # split-half reliability, primary-score rule
python scripts/s06_simulate.py              # Phase 2 power / minimum resamples
python scripts/s07_validity.py              # parsing, truncation, looping, eligibility audit
python scripts/s10_features.py              # boundary features (needs lm_head shard 4)
python scripts/s11_evaluate.py              # grouped nested evaluation, layer curve, lead time
python scripts/s08_memorisation.py --endpoint http://127.0.0.1:8000   # needs the vLLM server
python -m pytest -q tests
```

The Phase 2 generation that is running uses the earlier driver
(`research/redesign/phase2_generate.py`, on the GPU instance; see `docs/compute_budget.md`).
Its completions are scored by this study's `scripts/s12_phase2_score.py`. That scorer was
written and tested before any Phase 2 data were read, on a synthetic split-half index and a
unit test. Once generation finishes:

```
# on the instance: build the completion index
uv run --frozen python research/redesign/phase2_generate.py index --inputs $IN --run-dir $P2_RUN
# copy completions_index.parquet locally, then:
python scripts/s12_phase2_score.py --index completions_index.parquet \
  --inputs ~/preanchor-root/data/processed/redesign_phase2/inputs.parquet
```

## Findings

### Confirmed (executed and verified)

- **Exact reproduction:** the MVP labels (4,019 / 4,019, maximum difference 2.7e-15)
  reproduce exactly from reacquired, hash-verified rollouts.
- **KL instability:** with α = 1e-9, a 6% unseen answer alone contributes ≈ 1.35 nats.
  96% of MVP anchors are driven by such answers.
- **Split-half reliability (N ≈ 50 per half):**
  - No score passes: ICC 0.39–0.62, Spearman 0.35–0.59, κ(q90) 0.24–0.40, all FAIL.
  - The MVP KL is the least reliable, apart from `dacc`.
  - Selected primary: `fork_tv`.
- **Noise floor:** the median TV of 0.071 barely exceeds its no-effect floor of 0.065,
  and only 55% of sentences exceed their own floor.
- **Power:** at 100 resamples, Phase 2 is forecast to FAIL (P ≈ 0.98, driven by
  Spearman). About 200 resamples are needed for ICC ≥ 0.75, and about 400 for `jsd` to
  pass both criteria (P = 0.74).
- **Parsing:** v2 fixes four v1 defects. 0.35% of answers change. Bounded scores are
  insensitive to the answer-handling choices (ρ ≥ 0.97).
- **Eligibility:** the two MVP rules that used future information are replaced, with
  leakage tests.
- **Memorisation:** 1 of 20 problems flagged (1591), with no verbatim recall.
- **Phase 2 reliability replication (CONFIRMATORY, plan v2.1; 40,000 generations, n = 100 per fork):**
  - Primary `fork_tv`, replicate 1 vs 2: Spearman 0.445 [0.198, 0.617] and ICC 0.561 [0.228, 0.676], so the continuous form is **FAIL**.
  - The flag gives κ 0.659 [0.000, 0.855], so it is **INCONCLUSIVE**: the fixed q90 threshold flags only 6 and 3 of 190 sentences.
  - All five scores FAIL. The result is unchanged without problem 1591, and agreement with the released samples is similar (ICC about 0.62).
  - Details: `docs/phase2_results.md`.
- **Prediction (full run, 8 layers; nested problem-grouped CV over all 20 problems):**
  - Activations beat every baseline in point estimate: Spearman 0.23 vs 0.15 for
    `fork_tv`.
  - All 12 nested-comparison intervals include 0. One previously appeared to exclude it
    (+0.048 [+0.001, +0.093]), but that came from a non-nested comparator. Corrected, it is
    +0.036 [−0.002, +0.085] (`docs/evaluation_results.md`).
  - The layer curve peaks at 32–40, and layer 0 already gets AP 0.13 against a base rate
    of 0.11.
  - Lead 1 ≈ lead 0.
  - Noise ceiling ≈ 0.87.
  - Details: `docs/evaluation_results.md`.

### Assumptions

- The rule that a fork-i resample whose first sentence has cosine < 0.8 counts as
  "without S_i" is inherited.
- Spearman–Brown extrapolation to full N holds.
- The simulation takes the empirical distribution as truth, which makes it optimistic.
- The marker-token list.
- The $2.49/h rate.

### Proposed changes (not adopted into the MVP)

- Use a bounded score (`fork_tv` or `jsd`) with ≥ 200–400 resamples per fork.
- Use parser v2.
- Use pre-emission eligibility.
- Phase 3 on post-cutoff problems behind the memorisation gate.

### Unresolved

- Whether to raise resampling depth or change the target before Phase 3, after the Phase 2 FAIL.
- The undefined "14 problems".
- The 0.132 width reading in the Phase 3 sample size.
- The Phase 3 problem list.
- The billed GPU rate.
- A throughput pilot through this folder's own client: the GPU was occupied, and the
  measured benchmark came from the earlier driver.
- The Phase 2 launch preceded plan v2 and the memorisation report (deviation recorded in
  plan §13).
