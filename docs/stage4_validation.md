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
