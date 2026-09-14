# Data dictionary

Document table grain, field meaning, schema versions, provenance, eligibility and feature exclusions.

Stage interfaces are specified in SPEC §5. No data tables have been produced.
The 12 observable column names in `configs/mvp.yaml` map in order to SPEC §3.9;
`generated_token_count` counts tokens since the start of `<think>`, and
`prefix_context_fraction` divides prefix token count by 32768. Neither uses completed length.
