# Stage 8 plan and pre-production validation (2026-09-15)

Branch `codex/stage8`, from the `codex/stage7` merge `480031e`. Stage 8 fits the SPEC
§3.10 model table on the 12 training problems under four-fold problem-grouped CV and
selects the layer and regularisation from training CV only. CPU only
(`CUDA_VISIBLE_DEVICES=""`).

## Inputs (all from the finalized Stage 7 run `20260915T182306Z-3ab408d80a-7452f1c2`)

| Input | Hash | Use |
|---|---|---|
| `data/processed/stage7/features.parquet` | `f3227c386383…9a064d` | rows, split, 12 observables, targets, shard pointers |
| `activations/index.parquet` | `1c6c9c2fab7d…ebe5a` | identity of the shards |
| `data/processed/stage4/labels.parquet` | `fab9886e4789…a61b31` | recorded input of the Stage 7 run |
| `activations/units/*/<problem>_<kind>.safetensors` | per-unit `SUCCESS.json` | one site at a time, float16 → float32 |

Training partition: 2,650 eligible rows, 265 anchors (threshold 1.354981). Folds
(seed `derived_seed(20260914, "cv_folds")`): validation problems {1591, 2137},
{2189, 2238, 4682, 6596}, {3448, 3935, 4019}, {3360, 3550, 4605} with 17 / 48 / 40 / 160
validation anchors. Problem 3550 holds 156 training anchors, so fold 3 dominates the
positives; this is reported, not changed.

## Implementation

`src/preanchor/probes/`: `dataset.py` (table/run identity checks, row selection, site
loader), `folds.py`, `models.py` (pipelines, `Residualizer`, `Prevalence`), `metrics.py`,
`fit.py` (resumable units, aggregation), `select.py`, `validate.py`, `calibrate.py`
(skip rule only). CLI: `preanchor fit --activation-run-id <S7> [--resume --run-id <S8>]`,
`preanchor select --run-id <S8>`, `preanchor validate-probes --run-id <S8>`.
Tests: `tests/unit/test_probe_folds.py`, `tests/integration/test_probe_fit.py`,
`tests/scientific/test_split_isolation.py`, `test_pipeline_fit_scope.py`,
`test_probe_sanity.py`.

## Smoke evidence (real data, `configs/smoke.yaml`, two sites)

Run `20260915T191139Z-30a5e56f6f-5c6fdbe6`: 9 fit units in 80 s (activation load ≈ 37 s,
peak RSS 2.0 GB, 8 workers); `select` 85 s (residualised control, 10 refits, peak RSS
2.6 GB); `validate-probes` ok, 580 CV rows, 384,250 fold predictions, AP recomputation
difference 0.0, no non-converged fit. Per activation unit 9-33 s wall.

## Projection for production (`configs/mvp.yaml`, 49 sites)

MEASURED per unit above; EXTRAPOLATED: 5 baselines + 98 activation units ≈ 98 × 15 s +
load ≈ 25-35 min for `fit`, ≈ 2 min for `select`; peak RAM < 4 GB; outputs ≈ 100 units ×
0.6 MB + run tables ≈ 150 MB. Under one hour, so no SPEC §8 GPU projection applies; a
`resource_projection.json` is still written from the smoke figures at launch.

## Production sequence

1. Stage 7 MacBook `pull_verification` receipt present (user-run, other session).
2. `uv run --frozen pytest -q tests/unit tests/integration tests/scientific` at the launch
   commit; clean tree.
3. `tmux`: `preanchor fit --config configs/mvp.yaml --activation-run-id 20260915T182306Z-3ab408d80a-7452f1c2 --workers 8`
4. `preanchor select --config configs/mvp.yaml --run-id <S8>`
5. `preanchor validate-probes --config configs/mvp.yaml --run-id <S8>`; record
   `docs/stage8_validation.md`; back up for stage 8.
