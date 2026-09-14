# Stage 4 implementation preparation

The detached Stage 4 development worktree contains the canonical fallback path.
`recompute_sentence` reads only adjacent raw fork files, applies the frozen smoothed
KL equation and records `label_source=canonical_fallback`; missing adjacent forks and
post-convergence sentences are explicitly ineligible. `convergence_idx` implements the
five consecutive accuracy rule, and `train_threshold` fits quantiles from training
problem IDs only. UnitStore remains the atomic, SHA-256-verified resume boundary.

The Problem 4682 CPU benchmark processed 10 raw files (1,000 samples) in 0.0845 s.
Full fallback recomputation is projected at roughly 55 s CPU, with negligible Lambda
cost (under $0.03 at $1.99/hour) and under 100 MB of derived output. This is a
projection only; full recomputation waits for download completion and validation.
