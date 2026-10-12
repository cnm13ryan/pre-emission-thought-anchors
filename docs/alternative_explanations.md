# Alternative explanations

For every positive finding, record competing explanations and the experiment that tests each.

No empirical findings exist yet. Future entries must cover position, prefix length,
uncertainty, tokenisation artifacts and label noise; link each to its control artifact
and state whether the finding survives. A valid null remains reportable.

## Tokenisation artifact recorded before any finding (2026-09-15, Stage 6)

For 2,844 of the 6,474 sentences (1,930 eligible) the sentence begins after a single space
and the pinned tokenizer's full-trace token spans that space and the first word (for example
`" Let"`). Under SPEC §3.8 that token belongs to the target, so the pre-emission boundary is
the last token of the preceding sentence. The label-generating prefixes, both the released
resamples and the Stage 5 vLLM prompts (`full_cot[:char_start]`), tokenise to that same
prefix plus one trailing `" "` token in 2,816 rows (Stage 5 forks 6, 7, 12, 14 and 47 of
`4682:correct`). Consequently the importance label is conditioned on one more whitespace
token than the activation boundary sees. Any positive finding must be checked against this
artifact: compare probe performance between space-initial and newline-initial sentences
(the `overlapping_token` column of `activations/index.parquet`), and report whether the
signal survives within each stratum. The frozen boundary definition is not changed.

## Stage 7 dataset notes for future findings (2026-09-15)

- Prefix-length and position explanations are testable directly from the table: the
  primary features include `prefix_token_count` (61–12,356 tokens; maximum context
  fraction 0.377) and `sentence_idx` (0–388), and the completed-length quantities exist only
  as `diagnostic__relative_completed_trace_position`. Any activation-probe gain must be
  reported against the position and context baselines built from these columns.
- Uncertainty explanations: `next_token_entropy` spans 1.2e-10 to 2.90 nats and
  `next_token_max_probability` 0.159 to 1.0 across the 6,474 boundaries, so the uncertainty
  baseline is not degenerate; the residualised control removes these columns fold-locally.
- Tokenisation artifact: 2,844 rows have `overlapping_token = true` (space-initial
  sentences; see the Stage 6 entry above). The column is in the feature table for the
  stratified comparison.
- Numerical-precision artifact: the next-token statistics are float32 from float16-stored
  states; a bfloat16 head changes them by up to a few hundredths (Stage 6 diagnostic). A
  probe result that depends on that precision would be an artifact, testable by recomputing
  the uncertainty columns with the bfloat16 head from `logit_features.parquet` of the smoke
  runs.
## 2026-09-18: blog audit of the transition positive control

The site-44 activation-only transition probe exceeds the saved previous-stage-plus-position baseline by AP 0.0427 on validation and 0.0555 on test; exploratory paired problem-bootstrap intervals are positive on both four-problem splits. See `research/blog/evidence.json` for reproduced values and seeds.

- Position and prefix length: the stronger baseline reduces the apparent advantage substantially; test richer position controls and prefix-text predictors before claiming conditional information beyond all observable structure.
- Uncertainty: scalar entropy/mass features discard token identity. Compare the full next-token distribution and marker probabilities.
- Tokenisation and lexical preparation: 770/830 backtracking labels begin with rule-associated opening markers. Test preceding-token/K-token baselines and earlier token boundaries; stratify overlapping-token cases and harmonise generation/extraction prefixes.
- Label noise: deterministic stages are surface labels, not audited functional reasoning annotations. Audit independent functional labels. The separate anchor target has strong smoothing sensitivity; repeated counterfactual resampling is still needed to measure reliability.
- Lead time: the activation-only earlier-state result can reflect anticipation of the intervening stage. Oracle controls are diagnostic, and exploratory position-augmented lead-one arms include target-boundary fields unavailable at the earlier boundary. Fresh matched-boundary experiments are required.
