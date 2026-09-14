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
