# v3 follow-up: CPU results (Steps 0–2)

**Status: EXPLORATORY, POST HOC.** These analyses follow up the plan_v3 pilot (`dd94808`),
whose verdict `INCONCLUSIVE_QUALITY_GATE` stands. No pilot file was edited. Each task's
`run.py`, `results.json` and `summary.md` are under `analysis/v3_followup/<task>/`.

**Common conventions:**
- Intervals are problem-cluster bootstraps.
- Answer grouping is canonical (`src/arv2/canon.py`), with raw reported where it differs.
- Scores are rounded to 10 decimals before any rank statistic.

| Item | Task (commit) | Result | Resolved? | Consequence for the GPU plan |
|---|---|---|---|---|
| — | Step 0 inventory (`66576ef`) | The sampled sentence i is stored for r0, r1 and r2. Activations are raw: 48 layers at one boundary token per sentence; no span states, no attention. Fork windows are drawn by a seeded random rule (`research/redesign/phase2_select.py`). The probe-capacity job finished (`3c2048c`) | Yes | T6 is feasible. All-token states and attention exist only in the forward-pass arms |
| 6 | Step 1 canonicaliser (`b4fa8c8`) | 174 → 170 categories, all decimal-vs-fraction merges. Every Phase 2 and pilot number moves by at most 0.025. Same 7 real effects. The forced-readout `<OTHER>` rate is unchanged (77.8% → 77.7%). Phase 2 fork_tv Spearman carries about ±0.01 of floating-point tie noise (0.445 committed; 0.446 with ties rounded) | Yes: grouping is not a cause | Use the canonical grouper everywhere. No grouping arm |
| 5 | T1 (`d888f22`) | Released minus ours, accuracy: −0.109 [−0.121, −0.096] on the 2 incorrect traces (lower at 40 of 40 forks); −0.005 on the correct ones. Temperature, top_p and the 16,384 cap match; truncation is rare (0.44% / 0.67%) and explains 0.2% of the gap; length matching widens it. The released prefixes collapse every newline to a space (0 of 200 prompts byte-identical, 197 of 200 after normalising whitespace). The released engine is a hosted API, Novita by default, unknown | **No: UNEXPLAINED** quantitatively. Prompt format is the leading candidate | **Arm A5 included**: the exact released prompts are cached at `derived/v3_followup/t1_released_vs_ours/`. A5 can separate prompt format from engine; the provider itself cannot be reproduced |
| 1 | T2 (`58ec288`) | Split-half reliability, SB to full n, 1,000 splits. `tv_debiased`: 0.16 [0.13, 0.19] on released data (6,434 sentences); 0.18 [−0.03, 0.36] in Phase 2 (300 per fork), where the ICC of 0.78 comes entirely from the 3 problems holding the real effects. About 2,100 samples per fork are needed for 0.80. `fork_tv` reaches 0.81, but the floor alone reaches 0.97. The s14 "0.90 / 0.39" were √-ceilings; the reliabilities are 0.80 / 0.15 | Yes: the anchoring-specific label is not reliable at affordable n | Do not brute-force the continuous score. Measure the ceiling with nested sampling; treat anchors as rare tested events |
| 7 | T3 (`ac40bfa`) | Window rule written out exactly and reproduced. Anchor rate (TV permutation test with Besag–Clifford sequential stopping, BH 10% over 6,434 sentences): 2.04% [1.65, 2.47] overall; 2.77% before convergence; 0.17% after; 3.82% in the top entropy tercile; 3.62% for the deployable "before convergence ∩ top half of a 20-sample screen" (81% of anchors kept, 45% of sentences). The Phase 2 window was typical: 3.68%, 67th percentile of 1,000 rule seeds | Yes: selection does not explain the low rate | Fork selection = cheap screen + half by entropy, half random, with inverse-probability weights |
| 8 | T4 (`f12f4ba`) + s16 (`3c2048c`) | Difficulty does not explain the null. Partialling difficulty out leaves acc_level, entropy and p_own unchanged (0.16 → 0.19; 0.48 → 0.48; 0.46 → 0.47), and the signal is within problems. Effect targets stay ≤ 0.07 in every stratum; pairwise ranking is 0.50–0.52. Full-width (5,120-d) ridge vs PCA-128: +0.00 to +0.03. Activations never beat position for p_own or entropy (−0.14 within trace). The no-reasoning difficulty measure is degenerate (18 of 20 problems at 0) | Yes, for linear single-token probes and difficulty | Single-boundary-token linear probes are exhausted. The forward-pass arms target other tokens and positions (all-token, span pooling, attention, logit lens), not more capacity |
| 4 | T6 (`2c13899`) | **Signed importance:** R²_max ≈ 0, by theory (tower property: E_S[acc(h,S)] = acc(h)) and in data (0.03 [0.01, 0.05] released; ≈ 0 in Phase 2), so the signed-target probe nulls were guaranteed. **Unsigned importance** (canonical TV): R²_max 0.57 [0.38, 0.77] released, almost all within problem; \|Δacc\| 0.50 [0.37, 0.66]. But the estimator fails on weak-effect synthetics matched to the real effect size (debiased m(h) ≈ 0.002), so the unsigned ceiling is indicative only. Probes reach about 4% of true-score variance, against about 30% for an oracle on the same noisy target | Signed: **yes**. Unsigned: **no**, the estimate is unreliable at real effect sizes | Drop signed targets. The nested arm becomes the core: **20 problems × 20 positions, K = 8 (K = 4 suffices on released variance components), M = 200**, scored with an **unbiased importance** (squared-L2 or (Δacc)² with cross-half products), not TV minus a null |
| 10 | T7 (`e718ef8`) | N for a 95% width of 0.30: S2 (level readout) ≈ 44 problems; S3 ≈ 75; κ(q90) ≈ 81–97; H1a ≈ 137–158; H4-level ≈ 201–221. These are between-problem dominated, so more forks do not help. S4–S7, the AP differences, ICC and the `tv_debiased` split-half are within-problem dominated, trading forks against problems. All N are lower bounds, extrapolated from 10 problems | Yes, as a lower bound | **N = 50 at about 20 forks**; H1a and H4-level demoted to descriptive (they would need N ≈ 200) |
| 12 | T5 (`c1c9ca8`) | Without problem 1591, no preregistered decision and no qualitative conclusion changes. The only flip is a secondary cell, tv κ r2 vs r0, FAIL → INCONCLUSIVE (upper bound 0.565 → 0.608). Phase 2 fork_tv Spearman 0.444 → 0.441; tv_debiased r1v2 0.151 → 0.195; same 7 real effects; pilot H1a 0.096 → 0.175 | Yes for 1591. A leave-one-out sweep (for example without 2238, which holds 4 of the 7 effects) was not run | No exclusion. Generality is tested by arm A6 (second source) |

## What changed in the picture

1. **The probe nulls are now explained twice over.**
   - **Signed targets** cannot be predicted from history at all (T6, by construction).
   - **Effect labels** are mostly noise at affordable n (T2).
   - **Difficulty** is not the cause (T4), and neither is answer grouping (Step 1).
   - **Linear capacity** is not the limit either (s16, T4).
2. **One opening remains: unsigned importance.** Its magnitude may have a substantial
   ceiling (T6: about 0.5) that the probes reach only a small fraction of. But that ceiling
   cannot be trusted at the real effect size. Nested sampling with M ≈ 200 is the only way
   to settle it.
3. **Anchors are rare tested events.**
   - About 2% of all sentences, and 3–4% in uncertain, pre-convergence regions (T3).
   - In Phase 2 they concentrate in 3 problems (T2 and T5).
4. **The released-vs-ours gap is real and confined to the incorrect traces.** Its
   leading explanation is prompt formatting (T1). That needs a generation check, arm A5.

## Process notes

- **The summaries were written from the shell.** The file-writing tool refused `summary.md`
  for every Step 2 agent, so each wrote it with a shell heredoc. Every commit touches only
  its own task directory.
- **T1 and T6 used resources beyond the CPU.**
  - T1 made public read-only fetches: 200 released solution files, sha256-verified against
    the MVP manifest.
  - T6 computed MiniLM embeddings on the local Apple GPU (MPS).
- **Leftovers.** An untracked `__pycache__` remains in `t6_nested_ceiling/`.
