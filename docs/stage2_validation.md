# Stage 2 validation

CPU-only selective source acquisition is in progress; no scientific result or GPU admission.
Implementation based on `fbe40e1`. The exact generated allowlist contains 306 files,
339,390,844 bytes, all forty trace/label pairs and 145 correct-4682 raw forks.

Minimal inspection is retained under
`$PREANCHOR_ROOT/artifacts/bootstrap/stage2-minimal/`, including rejected partial units.
Raw upstream bytes are unchanged. Separate schema fixtures cover chunks-object versus
labeled-list layouts; upstream Git/LFS hashes and local SHA-256 are checked independently.

Unit/integration and Stage 2 source-scope tests: 60 passed (two existing Typer warnings).
Full detached smoke/kill/resume, scale validation and secondary backup are still pending.
Future scientific tests are not represented by this count.
