# Step 0: inventory

**Status:** EXPLORATORY, POST HOC (follow-up to the plan_v3 pilot, `dd94808`). Nothing here
changes a pilot file. Details are in `results.json`.

| Question | Answer |
|---|---|
| Do the resamples store the sampled sentence i? | **Yes.** Phase 2 r1/r2 (40,000 rows): `first_replacement_sentence` plus the full `raw_answer` continuation, with none missing. Released r0 (6,474 fork files): `chunk_resampled` plus boxes and answer, with none missing, but the full continuation text is not kept locally. |
| Which activations are stored? | **Raw** `resid` [n, 49, 5120] fp16: the embeddings plus all 48 layers, at the `block_post_pre_final_norm` site, for **one token per sentence** (the boundary token just before it). There is also `final_prev_position`. They cover 6,474 sentences in the 40 MVP traces, including every Phase 2 fork. PCA-128 exists only inside probe fits. **No** sentence-span or all-token states, and **no** attention. |
| Which code selects the fork windows? | `research/redesign/phase2_select.py` (parent repo), whose output `configs/phase2_forks.json` drives Phase 2 and the pilot. The rule is a **seeded random** pick of 10 of the 16 non-test problems, with a random trace kind, and a uniformly random start for a 20-fork window that ends before the convergence and final-answer sentences. It does not depend on importance. |
| What is the status of the probe-capacity job? | **Finished and committed** (`3c2048c`, `docs/probe_capacity.md`). The null holds across 108 probe settings. The best decoding of `p_own` is explained by position plus entropy. The `acc_level` null comes from sign cancellation between correct and incorrect traces. |

## Consequences for the tasks

- **T6** is feasible, because sentence i and the answer are stored for all three replicates.
- **T4: raw vs PCA-128.** This is feasible, since raw states are stored. s16 already went to PCA-512 (83% variance); full 5120-d ridge was skipped for runtime.
- **T4: what s16 already covers.** s16 tested the outcome-identity labels (`p_own`), which vary within a problem. T4 adds difficulty stratification and a full-width check.
- **Moved to Step 3.** Any span-pooled or all-token state, and attention, are outside these data and belong to the GPU forward-pass arms. No Step 2 task becomes infeasible.
- **T3.** The current window rule is random, so T3 must simulate the other selection criteria over the MVP resamples (100 per fork, all forks) rather than read them off Phase 2.
