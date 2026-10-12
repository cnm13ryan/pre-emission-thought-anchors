# Cleanup and hardening plan (branch `refactor/cleanup-and-hardening`)

Created 2026-10-12 from `main` at `55e5028` (PR #5 merged). The work is ordered by priority.

**Rules for this branch:**
- One work item per commit.
- No frozen MVP item changes (AGENTS.md §4).
- No GPU and no instance.
- Any change that alters a reported number is logged in `docs/decisions.md` with before
  and after values, and the affected doc gets a dated correction.
- `tests/unit` and `tests/integration`, and `anchor_reliability_v2/tests`, must pass on the
  CPU environment before each commit.

## A. Correctness and integrity (Codex review round 5, deferred from PR #5)

| # | Item | Where | May change numbers? |
|---|---|---|---|
| A1 | Fit the q90 flag cutoff inside each **inner** fold. Today it is fitted on the whole outer-training partition, so inner validation folds help set their own labels during model selection. | `anchor_reliability_v2/scripts/s11_evaluate.py` | **Yes.** Re-run s11 (about 45 min CPU) and update `evaluation_results.md` |
| A2 | Equal-N selection: take the first `common_n` **successful** samples per fork and replicate by rank, not `sample_idx < common_n`. | `anchor_reliability_v2/scripts/s12_phase2_score.py` | No on current data (all 100/100); verify |
| A3 | Validate cached forced-answer responses before reuse: parse them, compare the embedded request with the payload, and check a receipt. Otherwise quarantine and reacquire, as the s02 fork store does. | `anchor_reliability_v2/scripts/s18_forced_answer.py` | No; verify against the stored pilot |
| A4 | Make the blog evidence rebuildable from tracked artifacts. Persist the derived transition predictions instead of reading the ignored 126 MB activation arrays, and remove author-specific absolute paths. | `src/preanchor/report/blog_evidence.py`, `research/reasoning_stage/` | No; the rebuild must reproduce the committed metrics |

## B. Reproducibility and environment

| # | Item |
|---|---|
| B1 | Tests that need `torch`, `transformers` or `tokenizers` should skip cleanly on a CPU-only environment (`pytest.importorskip` or a `gpu` marker), so the CPU run is green and GPU-only tests are listed. Today: 17 failed and 23 errors from missing GPU extras. |
| B2 | Document recreating `.venv` (`uv sync --frozen`). The local venv's entry points still point at the repo's old path; this is local only, with no repo change beyond a note. |
| B3 | Before any GPU run, extend the hardware allow-list in `src/preanchor/labels/a6000.py` (H100 SXM, A100 SXM 80 GB) and add matching `configs/lambda/` profiles. The GPU-class change itself still needs owner approval. |

## C. Consolidate the study library (`anchor_reliability_v2/src/arv2`)

| # | Item |
|---|---|
| C1 | Move shared statistics out of scripts into `arv2`: Benjamini–Hochberg (copied in s13, s17, step1, T3 and T5), the per-sentence permutation test, the cross-half unbiased importance estimator (T6), `common_counts` (s12) and the TV noise floor. Add unit tests. |
| C2 | Point `scripts/` and `analysis/v3_followup/*/run.py` at the shared functions. Re-run each touched analysis and confirm byte-identical or numerically identical outputs. |
| C3 | Choose a licence for the public repo (owner decision; there is no LICENSE file). `vendor/thought-anchors` keeps its own MIT notice. |

## D. Lint and style (active code only)

About 420 ruff findings, concentrated in archived scripts and analysis code.

| # | Item |
|---|---|
| D1 | Review all 47 `B023` (function uses loop variable). Most are closures in bootstrap lambdas evaluated immediately; fix by binding defaults, and treat any that are not immediate as bugs. |
| D2 | `B905` (`zip` without `strict=`): add `strict=True` where lengths must match. |
| D3 | Mechanical fixes: E702/E701 (multiple statements per line), F401 (unused imports), I001 (import order), E501 (line length) in `src/`, `anchor_reliability_v2/src`, active `scripts/` and `apps/`. |
| D4 | Exclude `scripts/archive/*_obsolete.py` from lint in `pyproject.toml`, rather than editing archived code. |

## E. Documentation

| # | Item |
|---|---|
| E1 | Add one docs index (`anchor_reliability_v2/docs/README.md`) with each document's status: current, superseded or exploratory. Example: `situation_overview.md` is superseded by `plain_language_report.md`; plan drafts v1–v2 are superseded by v3. |
| E2 | Top-level `README.md`: point to the study, the plain-language report and the docs index. |

## Out of scope here

- Any GPU work or launch (the GPU plan awaits owner approval).
- New experiments or analyses.
- The ownership question in `PROJECT_PROVENANCE.md` (owner decision).
