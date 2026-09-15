# Upstream audit

Record upstream reuse, paper/released/legacy/canonical differences and the primary-label gate.

Stage 0 inspection (implementation based on commit `85a5604`):

| Repository | Required commit | Licence |
| --- | --- | --- |
| thought-anchors | b53ed8c75d3f6112f68adfaec9a13d4d708c442e | MIT; `LICENSE.md`, copyright 2025 interp-reasoning |
| thought-anchors.com | 67071a30bd2173c56f1772a26f6a9e6855170035 | MIT; `LICENSE.md`, copyright 2025 interp-reasoning |

Both are unchanged submodules. No upstream routines have been copied or wrapped yet.
Retain licence notices with future ports. Label parity and primary-field selection have
not been audited and are not established by the bootstrap tests.

Stage 2 source inspection (execution commit `700d348`):
`vendor/thought-anchors/selected_problems.json` exists at the pinned commit. It has
106 records, and its first twenty problem IDs exactly equal the frozen MVP set.
SHA-256: `9a89831a1bda3f880e560c4ee75908e6972305d7bce1717a377122854646481e`.
This corrects the earlier continuation record's failed search; it changes no selection.
The file remains unmodified and is included through the pinned submodule Git bundle.
Stage 2 independently implements acquisition; no upstream scientific routine is copied.

Stage 4 closure (commit 5e92130): the Problem 4682 released-label audit produced finite/nonnegative fraction 1.0 but Spearman 0.6217465713409153 versus canonical, below the frozen >=0.95 gate. Released labels therefore cannot be `importance_primary`; canonical smoothed KL is the frozen fallback. The fallback run 20260914T190027Z-stage4fallback-6e78475 independently verified 6,474 files and 20,996,423,633 bytes with zero checksum, missing, duplicate, or unresolved units (manifest SHA-256 caf3ce503ca0260db47301b6e98bae9fa61831c5de8be0782e970ebc2c9f9127). The source revision and requested paths were unchanged. The initial snapshot_download materialization failed at problem_6596/chunk_103/solutions.json; direct hf_hub_download recovered it. 2,600 pre-fix units were revalidated, one incomplete unit was quarantined and regenerated, and subsequent resumes reused valid units. Canonical labels contain 6,474 rows (6,434 eligible); training-only q90 is 1.0685654935597562.

### Support-unit optimization closure

Commit `6a9ae6e` is the active batched support implementation. The prior unreferenced per-sentence predecessor is retained under `scripts/archive/` for provenance. The CUDA path was compared with authoritative CPU/FP32 on 10,000 cosine comparisons: maximum absolute difference `2.980232238769531e-7`, median `1.1920928955078125e-7`, p99 `2.980232238769531e-7`, and zero `<0.8` classification or support-count disagreements. The pinned model/revision and threshold were unchanged. All 6,474 atomic support outputs validated; full invocation completed in 197.039 seconds. Compact raw-derived sensitivity inputs and train-only q80/q90/q95 plus K=4/K=5 masks are recorded in `data/processed/stage4/raw_sensitivity_inputs.parquet` and its provenance JSON.

## Fresh pilot answer parsing correction (2026-09-15; commit introducing this entry)

`labels/upstream_answers.py` ports `extract_boxed_answers` and `normalize_latex`
from pinned `utils.py`; MIT copyright and license are retained in the module.
Only typing/formatting changes: behavior, including first-box selection and
unterminated-box extraction, matches upstream `generate_rollouts.py`.
`labels/fresh.py` implements the frozen union-support canonical KL independently.
Legacy contrast excludes missing answers and smooths over the observed union.
Old `importance.py` and historical Stage 4 outputs remain intact; their support
normalization defect needs a separate historical audit. The Stage 5 runner never
uses the whole continuation as its answer category when constructing labels.

## 2026-09-15 corrected Stage 4 audit (run `20260915T120136Z-b1a2dd9ab0-3b77595c`)

The earlier "Stage 4 closure" and "Support-unit optimization closure" entries above
describe defective outputs (no dissimilarity filter, self-similarity support units) and
are superseded; those artifacts are marked pre-fix/non-canonical.

| Field | Definition now implemented | Evidence |
| --- | --- | --- |
| paper equation | KL(P(A \| dissimilar S_i replacements) ‖ P(A \| S_i)) | SPEC §3.6 |
| `importance_released` | dataset `counterfactual_importance_kl` | Stage 3 tables |
| `importance_legacy` | pinned `calculate_counterfactual_importance_kl`: alpha 1.0, `use_similar_chunks=True`, missing answers dropped, `normalize_latex` applied three times, local vocabulary, `p_unsmoothed * log(p/q)` | `tests/integration/test_legacy_parity.py` executes the pinned source as the oracle |
| `importance_canonical` | union categories incl. `<NO_ANSWER>`, one K, alpha 1e-9, dissimilar-only "without", fork i+1 "with" | `tests/unit/test_importance.py` hand calculations |

Observed differences on Problem 4682 (correct): released vs canonical Spearman 0.857966;
released vs legacy 0.744655; canonical vs legacy 0.680600. The released field is not
reproduced by the pinned defaults either, so its exact generation settings remain
unknown. Primary field: `importance_canonical` (gate failed). The pinned similarity
model is used at the frozen revision; the pinned repository loads it by name only.
