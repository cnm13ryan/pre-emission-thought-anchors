# Stage 7 validation — full activation and feature extraction (2026-09-15, H100 PCIe)

Branch `codex/stage7`, created from the `codex/stage6` merge commit `afd3714` (which
contains Stage 6 `542ee91` and the portable-backup work). Implementation commits `1cfa152`
(build-dataset, validate-features, tests) and `3c2d4ec` (mvp-only, fail-closed publication).
**Launch commit for every Stage 7 run: `3c2d4ec8588ed7e9bef554e62b52ca65f330a0c8`, clean
tree** (`dirty_diff_hash` is the empty-diff hash `8794212a…`). Persistent root
`/lambda/nfs/preanchor/preanchor` (virtiofs `8dd29228-…`); every run is under
`artifacts/runs/`. Stage 3–6 canonical artifacts were read only. Cost figures use
`PREANCHOR_HOURLY_RATE=2.49`. Stage 8 was not started.

Inputs (identical in every run, equal to the Stage 6 records): `stage3/trajectories.parquet`
`4fd5fbee…39913d`, `stage3/sentences.parquet` `83cf8c20…143cf8`, `stage4/labels.parquet`
`fab9886e…a61b31`; model and tokenizer revision `1df8507178afcc1bef68cd8c393f61a886323761`;
lock `1126f52a…`. `configs/mvp.yaml` resolved hash `c8414238…6ef4f`, scientific hash
`82e38884…1a948` (the `9bad4b36…` / `14e279c5…` pair that `docs/stage6_plan.md` and
`docs/stage6_validation.md` attribute to `configs/mvp.yaml` is in fact `configs/pilot.yaml`,
the Stage 5 configuration; no configuration file changed).

## Launch-time admission (all PASS before production)

| Gate | Status | Evidence |
|---|---|---|
| HEAD contains Stage 6 `542ee91` | PASS | `git merge-base --is-ancestor 542ee91 3c2d4ec` |
| Canonical Stage 3/4 inputs validate and hash as recorded | PASS | `load_inputs` in every run (schema sidecars, checksums, trace hashes, span slices, Stage 3/4 span agreement) |
| Stage 5 independent MacBook secondary backup | PASS | bundle `20260915T163333Z-d565605b…132a2`: `preanchor backup verify` rehashed 4,282 objects / 33,704,252 B, 0 missing, 0 corrupt; receipt `artifacts/portable/receipts/20260915T163431Z-pull_verification-d565605b9518-62c783c6.json` checksum valid, `outcome: secondary_copy_verified`, `manifest_sha256` equals the bundle manifest's hash and sidecar. The `backup.verified: false` field in run manifests is not authoritative |
| Production doctor (`configs/mvp.yaml`) | PASS | all checks ok, `gpu_stage_admitted: true`, H100 PCIe 81,559 MiB, driver 580.105.08, torch 2.7.0+cu126, model revision resolves to the pinned SHA, W&B offline fallback |
| CPU tests at the launch commit | PASS | 250 passed, 1 skipped (`tests/unit`, `tests/integration`, `tests/scientific`) |
| GPU scientific gate `tests/scientific/test_preemission_invariance.py` | PASS | pinned model, run inside both smoke sessions (67–69 s) |
| Stage 7 smoke path at the launch commit | PASS | run `20260915T181921Z-00f7ba7cb0-ec063f56` (below) |
| Resource projection newer than the current commit | PASS | `stages/stage6/resource_projection.json` of the smoke run, `code_commit = 3c2d4ec`: 40 units, 6,474 rows, 187,579 tokens, 215 s, 3.31 GB, $0.15 |
| Authorised runtime/budget | PASS | `PREANCHOR_MAX_RUNTIME_HOURS` and `PREANCHOR_MAX_COST` are unset in `.env`; the projection (0.06 GPU-h, $0.15) is two orders of magnitude below the Stage 5 stop rules (10 h / $20) and the user authorised Stage 7 explicitly |
| NFS capacity | PASS | 1.94 PB available versus the 6.63 GB disk-growth upper estimate (2× output) |
| GPU idle before launch | PASS | `nvidia-smi` 0 MiB, no compute processes, no vLLM |
| Frozen scientific settings unchanged | PASS | `configs/mvp.yaml` untouched since `69c05c9`; 49 sites, boundary `target_token_start − 1`, float16 storage, bf16 SDPA inference |

## Runs

| Run | Purpose | Commit | Result |
|---|---|---|---|
| `20260915T181227Z-6ada5f6497-cb77a75e` | smoke at `1cfa152` (330:correct, 16 rows) | `1cfa152` | complete; primary gate 0.0; logit identity exact; build-dataset and validate-features pass; **published its 16-row table to `data/processed/stage7/`** (design flaw fixed in `3c2d4ec`; the files were relocated, not deleted, by the production build: `data/processed/stage7/superseded/20260915T181227Z-6ada5f6497-cb77a75e/`) |
| `20260915T181921Z-00f7ba7cb0-ec063f56` | smoke at the launch commit | `3c2d4ec` | complete; 16 rows; primary gate max abs 0.0; 14/16 truncated-prefix diagnostics within 0.02 (max 4.5, as in every Stage 6 run); logit identity exact; stored fp16 vs fresh bf16 max abs 3.0e-8; shard `987ce4ab…656fd` byte-identical to the four Stage 6 runs; build-dataset (no publication) and validate-features (16/16 rows recomputed, max abs diff 0.0) pass; projection regenerated |
| `20260915T182306Z-3ab408d80a-7452f1c2` | **SPEC Stage 7 production** | `3c2d4ec` | complete; 40 units, 6,474 rows; `validate-activations`, `build-dataset`, `validate-features` pass; no failures, retries, resumes or quarantines |

Both production stages ran detached in `tmux` (`stage7-prod`) with a 5-second resource
monitor (`stage7-monitor`) sampling `nvidia-smi`, NFS usage and `heartbeat.json`.

## Production outputs (`artifacts/runs/20260915T182306Z-3ab408d80a-7452f1c2/`)

- `activations/units/<unit_key>/<problem>_<kind>.safetensors` × 40, each with
  `shard_meta.json` and `SUCCESS.json`; `resid` `[n, 49, 5120]` float16 and
  `final_prev_position` `[n, 5120]` float16 (`final_prev_position_available` true for all
  6,474 rows; minimum boundary index 60). Total 3,314,695,040 B (512,001 B/row).
- `activations/index.parquet`: 6,474 rows, 6,474 unique `sentence_id`s, equal to the Stage 4
  label set; `n_layers 49`, `d_model 5120`, `dtype float16`,
  `activation_site block_post_pre_final_norm`; every `trace_sha256` equals the Stage 3
  `full_cot_sha256`; `labels_sha256 = fab9886e…`. SHA-256 `1c6c9c2fab7d9e89d76cbc2d5d0990f7391672743d3c52f9613622440ebe6e5a`.
- `features/features.parquet` (6,474 rows × 40 columns) = published
  `data/processed/stage7/features.parquet`, SHA-256
  `f3227c386383701741aa284cc3bbf6b7c3205f2de9593616575bf1187e9a064d`; `features/feature_audit.json`,
  `features/feature_validation.json`; split rows train 4,030 / validation 1,363 / test 1,081;
  eligible 4,019; anchors 344 (equal to the Stage 4 summary).
- `stages/stage7/summary.json`, `resource_projection.json` (projection recomputed from the
  production measurements themselves), `attempts/…`, `heartbeat.json` (`complete 40/40`),
  W&B offline runs `qdt2taaz` (extraction) and `asf02wtv` (features) under `logs/wandb/`.

Shard identity: `3935:incorrect` (389 rows, 13,087 tokens) hashes
`472bc53043c783a4e60e6e355da89a81cb18b82e77f366a8058c26c58e173824` in production and in the
Stage 6 envelope run from commit `f9dd294`: byte-identical across commits and sessions.
`330:correct` hashes `10aabf1b…` in production (all 26 rows) versus `987ce4ab…` in the
16-row smoke shards, as expected for a different row set.

## Validation gates (finalized run)

| Check | Result |
|---|---|
| `validate-activations`: every unit's `SUCCESS.json` hashes, shard checksum and metadata, index schema and coverage, index rows against shard boundaries | PASS, 40 units, 6,474 rows |
| Independent recount: 40 unit directories with `SUCCESS.json`; safetensors headers `(49, 5120)` / `(5120,)` F16 in all 40 shards; row sum 6,474; byte sum 3,314,695,040 | PASS |
| Coverage: index ids = feature ids = label ids (6,474, no duplicates) | PASS |
| `validate-features`: hash equals audit and manifest, published copy byte-identical, allow-list equals `features.observables`, no excluded name outside `diagnostic__`, all primary/diagnostic values finite, definitional invariants (prefix = boundary + 1, log1p, fraction, entropy in [0, ln V], probability ordering), joins to index and labels column by column, split from the frozen manifest, **256 seeded rows recomputed from the shards in float32: max abs diff 0.0** | PASS |
| Same-shape future-token mutation (primary causal gate, amendment `20260915T170000Z`) | PASS on the launch-commit smoke (max abs 0.0 at every site); production replays are bitwise identical to the Stage 6 envelope replay, which passed the gate at all 389 boundaries |
| Truncated-prefix replay (diagnostic, never a gate) | recorded: 14/16 within 0.02, max 4.5 at prefixes 70 and 86 tokens, unchanged from Stage 6 |
| Logit identity `lm_head(final_norm(layer48))` | exact (max abs 0.0, argmax equal) |
| Canonical input identities in the manifest | equal to the Stage 6 records (above) |
| No `failure.json`, no `stages/stage7/failures/`, no `activations/units/quarantine/` | confirmed |

## Resources: projection versus measurement

| Quantity | Launch projection (smoke, `3c2d4ec`) | Production measurement |
|---|---|---|
| Units / rows / tokens | 40 / 6,474 / 187,579 | 40 / 6,474 / 187,579 |
| Model load | 45.0 s | 63.0 s |
| Forward throughput | 3,077 tok/s (one cold 2,973-token forward) | 6,436 tok/s (29.1 s of forward time; per unit 0.24–2.95 s) |
| Peak VRAM allocated | 28,568 MiB | 29,973 MiB (`3935:incorrect`); `nvidia-smi` sampled maximum 34,391 MiB used including the CUDA context and cache |
| Peak host RSS | – | 9.5 GB (extraction), 4.1 GB (build-dataset) |
| Extraction attempt wall (load + 40 units + index + projection) | 215 s | 322 s (`/usr/bin/time` 5:28 for the process) |
| build-dataset | – | 61 s on `cuda:0` (float32 norm + `lm_head` from the pinned shards, no decoder) |
| validate-features (256-row recomputation) | – | 110 s |
| Output bytes | 3,314,755,977 | 3,314,695,040 shards + 0.85 MB index/features/audits; NFS used grew by 3.32 GB (72.72 → 76.04 GB) |
| Cost at 2.49 USD/h | $0.149 (extraction only) | extraction $0.22; whole pipeline 18:23:04–18:33:41 UTC (637 s) ≈ $0.44; the two smoke sessions ≈ $0.24 |

The projection's conservative doubling of the per-unit forward time and its cold single
forward made it pessimistic on throughput and optimistic on model load; the extraction wall
exceeded it by 50% (322 s versus 215 s), dominated by the 63 s load and per-unit NFS
writes, while GPU compute was 29 s. After completion `nvidia-smi` reported 0 MiB used and
the tmux sessions were closed.

## Observable definitions carried into the dataset

See `docs/data_dictionary.md`. Two definitional decisions (recorded in `docs/decisions.md`):
next-token statistics use the pinned `model.norm.weight` and `lm_head.weight` upcast to
float32 (the Stage 6 diagnostic `logit_features.parquet` applied them in bfloat16; on the
smoke rows the two differ by up to 0.034 in `next_token_max_probability` and 0.053 in the
first-target log-probability, which is why the Stage 7 table is the canonical one), and
`previous_sentence_tokens` is the token span `[target_token_start, target_token_end)` of
`S_{i−1}` (the Stage 6 diagnostic used the distance between successive sentence starts).

## Tests added (all CPU, tiny random Qwen2)

`tests/unit/test_observables.py` (allow-list equals the frozen config and rejects every
excluded name outside `diagnostic__`, float32 head reproduces `lm_head(norm(h))`, statistic
consistency, prompt token count from full-trace offsets, feature definitions, schema and
null enforcement) and `tests/integration/test_build_dataset.py` (build → validate on
synthetic Stage 3/4 tables, joins and split, deterministic identity across two extraction
runs, publication policy: mvp-only, canonical never overwritten, stale smoke publication
relocated; tampering detection through hash, published-copy and recomputation checks;
same-prefix/different-future invariance of every primary column with the completed-length
diagnostic changing).

## Unrelated observations (left unchanged)

- `tests/unit/test_support_resume.py::test_sentence_unit_kill_resume_preserves_completed_and_recomputes_partial`
  failed once while another session's pytest was running and passed on its own; the full
  suite passed twice afterwards.
- `ruff check .` reports pre-existing violations in `src/preanchor/labels/convergence.py`,
  `src/preanchor/labels/threshold.py` and `scripts/archive/`.
