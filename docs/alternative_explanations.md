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
