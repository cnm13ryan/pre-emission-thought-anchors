# Stage 11 plain-language presentation review — 2026-09-16

This is a presentation-only revision on `codex/stage11`, based on
`462c1ebf9f9a84ea82d7895ff3b279ab4d2f2ef5`. No scientific choices, calculations,
selection, datasets, or Stage 3–10 code/artifacts were changed. Existing reports
were retained. New rendered outputs have new presentation hashes; source hashes
and the recorded freeze are preserved in full.

## Deliverable

The rebuilt report and nine figures are at
`/Users/cnm13ryan/preanchor-stage11-readable-report/`:

- `reports/mvp_result.md` and `reports/mvp_result.html`;
- `report.json` SHA-256
  `4edbd6e5da6ddc255468fb49d4f4b8da2fba94273b60184e3aad0a71129386ff`;
- figures with standalone explanations, separate embed HTML, exact Plotly JSON,
  PNG exports, and checksum sidecars.

The Streamlit app builds the same explanations from the verified saved sources,
without writing a report or opening model payloads. Settings, original data,
full hashes, and probability-error metrics remain in expandable details. The
trajectory inspector explains counterfactual importance before displaying scores.
It distinguishes measured importance, stored label variants, and predicted chances.

## What did we learn?

1. Test AUCPR, a score for finding positive cases, is 0.0941, compared with the
   no-skill base rate 0.0363 (25 of 689 sentences). The descriptive interval is
   approximately 0.036–0.162.
2. The observed gain over combined simple clues is 0.0368, with an interval of
   approximately −0.051–0.081. It crosses zero: these data do not clearly establish
   that internal information genuinely improves on that simpler comparison.
3. Sixteen of 100 shuffled-label runs match or exceed the observed test AUCPR
   (0.0941; base rate 0.0363). Chance under this shuffle scheme remains plausible.
4. This is an exploratory held-out engineering test on four public problems, in
   a study of one model, one dataset and twenty problems. It does not establish
   generalization, a causal mechanism inside the predictor, or a safety monitor.

## Interpretation below each figure

Every major figure has six explicit prompts: question, no-skill expectation,
observation, uncertainty, conclusion, and limits. The same text appears below the
figure in the report, standalone figure HTML, and Streamlit. Interpretations are
constructed from saved values, including the direction of improvement intervals;
there are no hardcoded claims that every interval crosses zero.

| Figure | Plain-language reading |
|---|---|
| ROC / precision-recall | Can the predictor distinguish high-importance sentences? The observed ranking is above the no-skill reference, but that alone does not establish added value over simple clues. |
| Calibration | Do predicted chances match observed frequencies? Better ranking does not guarantee trustworthy probabilities; the constant forecast has lower test Brier error. |
| Layers | Where is prediction useful inside the model? Layer 44 was selected on training problems; no paired layer-comparison interval establishes it as uniquely special. |
| Baselines | Does internal information add to simple clues? The primary gain interval crosses zero. Sentence position alone also scores higher in this sample. |
| Lead-time probability boxes | How do predicted chances change one sentence earlier? These boxes show outputs, not accuracy. |
| Lead-time performance | Can useful prediction happen earlier? On the matching test rows, one-sentence-earlier AUCPR is 0.0319 (base rate 0.0322), with gain −0.0145 and interval −0.070–0.032. Validation does not show the same drop. |
| Threshold sensitivity | Does a different definition of high importance change the result? The observed gain becomes negative at the stricter cutoff; both broader and stricter comparisons have gain intervals crossing zero. |
| Position / uncertainty groups | Could simpler properties explain the pattern? Gains vary in direction and one group is undefined; sparse groups cannot rule out alternative explanations. |
| Shuffled-label comparison | Could this score arise by chance under the specified shuffle? Sixteen of 100 shuffled fits match or exceed it; this is not the probability that the research hypothesis is false. |

## Where evidence is insufficient

- Four test problems support descriptive intervals, not strong general claims.
- No paired uncertainty interval comparing layers is available in the layer view.
- No per-bin calibration intervals are stored in this view; some groups are small.
- Position groups can have no positives; subgroup patterns differ between test and validation.
- Only two prediction times were tested; the earliest useful time is not identified.
- The shuffle preserves problem, trajectory type and position groups, and cannot
  test all possible explanations. Single-class groups stay fixed.
- Public-data reuse, label noise and tokenization explanations remain limitations.

## Validation

- `CUDA_VISIBLE_DEVICES='' uv run pytest -q tests/integration/test_report_rebuild.py`:
  **6 passed**. Includes deterministic rebuilds, read-only inputs, tamper rejection,
  exporter compatibility, exact numerical regression, interval-language branches,
  and unchanged control coordinates/error bars with consistent subgroup colors.
- `tests/fixtures/stage11_presentation_original.json` records original numerical
  report fields and plotted coordinates from the pre-edit implementation at the
  commit above, using the existing synthetic fixture. The regression compares
  exact values, not rounded display strings or approximate tolerances.
- Real-data comparison against a pre-edit rebuild: **all original report fields
  identical**, excluding the intentionally revised figure list and figure hashes;
  **all five original figures' coordinates identical**; **all four restyled
  Stage 10 figures' coordinates/error bars identical** to the unchanged Stage 10
  builders; **all 12 input artifact hashes identical before and after**.
- Streamlit `AppTest`: **all five pages passed**, including changing problem,
  trace type, and sentence. Read-only evidence loaders remain checksum verified.
- `ruff check` on changed presentation/test files and `git diff --check`: passed.
  Git diff scope: Stage 11 report/app code, its regression fixtures/tests, this
  validation note. Stage 3–10 implementations and frozen configs are unchanged.
- Visual checks: rendered chart PNGs inspected, including calibration, ROC/PR,
  layers, threshold, subgroup, lead-time, baseline and permutation panels. An overlapping base-rate annotation and
  inconsistent subgroup legend colors were corrected without changing data.
- Browser integration could not start (tool bootstrap error: missing
  `sandboxPolicy`). Full browser/narrow-screen visual acceptance is **not claimed**.
- Full unit/integration command initially stopped during collection because
  PyTorch was absent. With the two collection-blocking modules excluded:
  **277 passed, 17 failed, 23 errors**. Failures/errors involve missing `torch`,
  `tokenizers`, or `transformers`, including a subprocess test that cannot reach
  its expected marker. Dependencies and unrelated stage code were not changed.
  This is not a claim that the entire repository suite passed.

No deployment, publication, new evaluation, frozen selection change, or write to
an existing persistent artifact was performed.
