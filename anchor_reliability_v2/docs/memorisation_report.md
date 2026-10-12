# Memorisation / contamination report

**Status:** EXECUTED on 2026-10-09 with `scripts/s08_memorisation.py`, against the pinned
vLLM server (H100 instance, reached through an SSH tunnel). Raw request/response pairs are
under `$ARV2_DATA/generations/memorisation/`. The full table is
`results/s08_memorisation.csv`.

**Protocol (plan §10):**
1. **No-reasoning accuracy:** the think block is closed immediately and `\boxed{` is
   forced. 20 samples at temperature 0.6, max 32 tokens.
2. **Verbatim continuation:** the first 50% of the problem's words, then 64 greedy tokens.

A problem is **flagged** if accuracy ≥ 0.5 or the next 10 words match exactly.

## Result

| Problem | Ground truth | No-reasoning accuracy | Modal answer | Next-10 exact | Edit similarity | Flag |
|---:|---|---:|---|:---:|---:|:---:|
| **1591** | 6.17 | **0.80** | 6.17 | no | 0.29 | **FLAGGED** |
| 4019 | 4 | 0.15 | 2 | no | 0.35 | |
| the other 18 | | 0.00 | wrong | no | 0.03–0.48 | |

- **Flagged: problem 1591**, one of the 20. Without reasoning, the model gives the
  correct 6.17 in 16 of 20 samples. That is consistent with memorisation, but the
  problem may also be easy enough to answer directly; the check cannot tell which.
- No problem's text is reproduced verbatim. The highest edit similarity is 0.48
  (problem 2236).

## Limitations

- **Problem 330** has only 2 words outside its formula, so its verbatim test compares a
  single word and is nearly uninformative.
- Low no-reasoning accuracy does not rule out contamination of the *reasoning* traces. The
  test only detects answer recall and text recall.
- The task asked for "all 14 mathematics problems". No 14-problem set exists in the MVP
  (there are 20 problems: 12 / 4 / 4), so all 20 were tested. The intended 14 need
  clarification.

## Use

- Problem 1591 is a training-split problem, and it **is** in the Phase 2 selection
  (`1591:correct`, forks 2–21; `configs/phase2_forks.json`).
- Phase 2 measures run-to-run reproducibility, which memorisation does not invalidate. A
  memorised answer may still compress that problem's answer distributions toward the
  correct value.
- The Phase 2 report therefore adds a leave-1591-out sensitivity. This is a reporting
  addition fixed now, before any Phase 2 data are read. It is not a change to the primary
  test.
- For Phase 3, the same protocol is applied to each new problem before generation, and
  flagged problems are excluded (`docs/evaluation_design.md` §B).
