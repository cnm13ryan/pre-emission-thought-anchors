# Stage 4 validation

The required Problem 4682 audit ran against the downloaded correct trajectory and all
145 released forks. Among 144 rows with an adjacent next fork, 100% of released values
were finite and nonnegative. The canonical smoothed KL recomputation had Spearman
correlation **0.6217465713409153** with released values, below the frozen 0.95 gate.

Therefore `importance_released` cannot be used as the primary label. Per SPEC §3.6,
the fallback is to fetch raw released resamples for all 40 trajectories and recompute
canonical labels. Discovery shows 6,474 raw `chunk_*/solutions.json` files totaling
20,996,423,633 bytes (20.996 GB). Using the measured source-fetch throughput gives a
projection above one hour. The required runtime, budget and hourly-rate approval is
pending; no fallback download has begun.

Audit artifact: `/lambda/nfs/preanchor/preanchor/data/processed/stage4/upstream_audit.json`.

Fallback verification run `20260914T190027Z-stage4fallback-6e78475` independently
verified 6,474/6,474 source files (20,996,423,633 bytes), with zero checksum,
missing, duplicate, or unresolved units after removal of the superseded empty
quarantine directory. The final source manifest SHA-256 is
`caf3ce503ca0260db47301b6e98bae9fa61831c5de8be0782e970ebc2c9f9127`.
Canonical fallback labels contain 6,474 rows, 6,434 eligible rows, and the
training-only q90 threshold is 1.0685654935597562.

Support closure: 6,474/6,474 sentence support outputs validated; CUDA/CPU parity had 10,000 comparisons and zero classification/support/NO_ANSWER disagreements. Full batched invocation wall time was 197.039 s. Sensitivity artifact rows: 6,474; q80=0.6147450059621669, q90=1.0685654935597562, q95=1.490691200555212; K=4 eligible rows=4,489 and K=5 eligible rows=4,627. Resume tests passed (2).

## 2026-09-15 regeneration after remediation

Superseded above: the 0.6217 Spearman, the 6,434 eligible rows, q90 1.0686, and the
support/parity closure were produced by code that omitted the dissimilarity filter and
compared sentences with themselves.

- Raw acquisition run `20260915T113709Z-734a9a5a41-5c37eb5c`: 6,474 files,
  20,996,423,633 bytes, six workers, 750 s, all upstream Git blob hashes verified;
  identity-free re-validation passed (`validate-raw-resamples --no-require-identity`).
- Audit run `20260915T120136Z-b1a2dd9ab0-3b77595c` (`stages/stage4/upstream_audit.*`):
  145 rows, 144 with adjacent fork, 126 compared; finite fraction 1.0; Spearman
  released vs canonical 0.857966, vs legacy 0.744655; gate failed; primary canonical.
- Label build (same run, `stages/stage4/labels.parquet`, SHA-256
  `2049a33f6ba3fc8ab51114ad3108bd8f8340e295f53470fab0e3d4e2a163d763`, published to
  `data/processed/stage4/`): 6,474 rows, 4,019 eligible, 3,902 K=4 eligible; q80
  0.816651, q90 1.354981, q95 1.827347; 344 anchors; similarity on CUDA in float32 with
  float64 dot products, fingerprinted per sentence; 368 s wall.
- Tests: 181 CPU tests pass (unit, integration, scientific) including the SPEC §10.3
  checks; ruff clean on every changed file. Production and CPU doctors pass.
- Pending: stage backup snapshot and MacBook secondary verification for the new runs.
