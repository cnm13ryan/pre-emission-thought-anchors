# Decisions

Record every result-affecting fact, assumption, choice and interpretation with commit and rejected alternative.

Entries below are based on repository commit `85a5604`; their implementation commit is
the Stage 0 bootstrap commit containing this file (`git log --follow docs/decisions.md`).

| Tag | Decision/evidence | Rejected alternative |
| --- | --- | --- |
| fact | User directs code checkout to `/home/ubuntu/preanchor-repo`; persistent state remains `/lambda/nfs/preanchor/preanchor`. | Treating the ephemeral checkout as authoritative experiment storage. |
| fact | `nvidia-smi` fails inside Codex sandbox; user confirms it works in normal host SSH. | Diagnosing a host driver failure from the sandbox result. |
| choice | Stage 0 CPU validation uses Python 3.11, with the prescribed GPU packages in the `gpu` extra and cu126 torch source. All dependencies are resolved in one `uv.lock`. | Installing GPU packages or admitting GPU stages during CPU bootstrap. |
| choice | Smoke and pilot configs inherit all scientific choices from MVP, changing only execution scope. Splits resolve into the hashed configuration. | Copying divergent science settings into separate configs or hashing only a split filename. |
| interpretation | YAML keys give explicit names to the SPEC §3.9 observable fields; no field is added to its allow-list. | Adding target-token probabilities or completed-length fields to primary inputs. |
| choice | `doctor --cpu-only` reports unchecked production prerequisites and cannot grant GPU admission. | Treating passing skeleton tests as hardware admission. |
| fact | Both pinned upstream repositories carry MIT notices; no source routine has been reused at Stage 0. | Assuming licence availability without reading the pinned notices. |
| choice | User chose MacBook-initiated SSH pulls into `~/preanchor-backups/` for the second durable copy. Stage 0 bundles only committed source, notices and validation documentation. | Pushing to the MacBook, uploading model caches/secrets, or claiming verification before the local receipt. |

No research results, model selection or scientific amendments have occurred.

Follow-up to bootstrap commit `2852b60` (`git log --follow src/preanchor/storage.py`):
**fact:** the designated persistent filesystem is exposed through writable `virtiofs`
under escalation. **choice:** fix the overly narrow NFS-type guard while retaining
dedicated-mount/path/write/checksum requirements. Rejected alternatives: treating the
root disk as durable, bypassing mount validation, or misreporting the virtual mount
as native NFS. The rejected attempt wrote no snapshot; science and config are unchanged.

Stage 1 (base commit `ad5758b`; implementation commit is the commit adding `manifest.py`):

- **fact:** the user supplied a MacBook verification receipt for all six Stage 0 files,
  manifest SHA-256 `dcd5417acca481bce61e1e50fbf79d2305f3f7f12b879ffb8877a03615f9c22c`,
  verified at `2026-09-14T17:37:27.331293+00:00`. Rechecked all persistent files and saved
  the receipt beside the snapshot; alternative rejected: inferring secondary verification
  from the existence of the first copy.
- **choice:** work-unit completion requires an immutable checksum manifest; incomplete or
  corrupt directories are renamed to quarantine, never deleted. OS advisory locks prevent
  concurrent writers and are released after abrupt termination. Alternative rejected:
  trusting file existence or rewriting valid completed samples.
- **choice:** run manifests have immutable checksummed revisions and an atomic current
  view; recovery validates that view against its revision if its sidecar publication was
  interrupted. Live heartbeat is replaceable metadata. Alternative rejected: accepting
  an unchecked current view or losing provenance through in-place edits.
- **choice:** run resume requires identical resolved config, code/diff, lock and input
  hashes. Artifact cache identities exclude tracking presentation but include all
  scientific inputs. Alternative rejected: silently mixing outputs from different runs.
- **choice:** Stage 1 smoke uses three tiny synthetic CPU units and tests both SIGTERM
  and SIGKILL. W&B offline files stay inside the run for backup; automatic hardware/code
  collection is disabled until the user's GPU-access gate is resolved. Explicit CPU RAM,
  rate and elapsed-time fields remain available. Alternative rejected: model inference
  or implicit GPU admission during I/O validation.
- **interpretation:** implementation-validation runs are not research experiments or
  evidence about probe performance. No Stage 1 artifact is a test metric or anchor label.
- **fact (after `b002449`):** the persistent smoke preserved completed hashes/mtimes
  through SIGTERM and SIGKILL, quarantined two partial units and completed a no-op resume.
  Full evidence is in `docs/stage1_validation.md`. The status reader initially called an
  old completed heartbeat stale; it now limits that flag to running work, with a regression
  test. Alternative rejected: treating normal completion as a stalled-job diagnosis.

Stage 2 (base commit `fbe40e1`; implementation commit adds `data/stage.py`):

- **fact:** the Stage 1 MacBook receipt verified 122 files, manifest
  `7dc74cfa250e5a4aded798faaf327819bff503cf7f13d68314578ffed07940f5` at
  `2026-09-14T18:10:50.343230+00:00`. Receipt and local recheck complete the gate.
  Rejected alternative: leaving the historical pending-backup status as current.
- **fact:** pinned discovery resolves 306 files, 339,390,844 bytes: forty each of
  base_solution, chunks_labeled, chunks and problem; README; 145 correct-4682 raw forks.
  Exact paths, source Git/LFS hashes and sizes are frozen in `configs/locks/dataset_*`.
  Rejected alternative: whole-repository, combined Parquet, other-model or variant downloads.
- **choice:** retain `chunks.json` and `problem.json` alongside primary trace/label files
  to preserve exact source segmentation and problem metadata for Stage 3. These add small
  auxiliary source files without changing the forty trajectories or scientific choices.
- **fact:** `selected_problems.json` was absent from the inspected dataset root and selected
  trajectory directories. A subsequent filesystem search found it in the pinned source
  checkout (106 records); its first twenty IDs exactly match the frozen set. The earlier
  handoff assertion of absence from the checkout was incorrect. No selection changed.
- **fact:** Hub 0.33.4 exports RepoFile/RepoFolder in `huggingface_hub.hf_api`, not the package
  root. `chunks.json` is an object containing string chunks; `chunks_labeled.json` is a
  list of objects. Minimal raw inspection exposed both draft assumptions before scale.
  Rejected alternative: dependency upgrades or transforming raw upstream bytes.
- **choice:** source files use exact singleton `snapshot_download` allowlists with revision
  pins, persistent HF cache and independent Git-blob/LFS checks followed by SHA-256 units.
  Main-loop heartbeats poll network worker completion. Graceful cancellation retains
  partial units; SIGKILL recovery quarantines them. Valid units are never rewritten.
- **interpretation:** Stage 2 schema/coverage tests are the applicable scientific gate
  for selective acquisition. Later pre-emission, label-audit and probe gates remain
  unimplemented and unclaimed. Downloading labels is not inspecting test metrics.
- **fact:** first uncached raw fork (3,302,151 bytes) took 0.778 s; the first base file
  took 0.643 s including singleton snapshot overhead. The 339 MB scope is expected well
  under one hour. A current-commit smoke measurement precedes the complete fetch; if its
  conservative projection reaches one hour, stop for the required cost/runtime gate.
  No GPU or model weights are needed; hourly GPU rate/budget remain unspecified.

- **fact (execution commit `700d348`):** the persistent five-file smoke passed SIGTERM,
  SIGKILL, partial quarantine and unchanged hashes/mtimes. The full 306-file fetch and
  no-op resume passed. A final error-path review found that a worker TimeoutError could
  be mistaken for a polling timeout; propagate completed-future exceptions, covered by
  a failure/resume regression. No download timeout occurred in the completed run.
  Rejected alternative: spinning indefinitely after a failed network request.

Stage 3/4 (working tree after `060f6c0`):

- **fact:** exact sequential import aligned 40 trajectories and 6,474 released chunks
  with zero unresolved spans. Outputs use checksummed Parquet and preserve raw source
  bytes. Rejected alternative: fuzzy/global replacement or reconstructing traces.
- **fact:** Problem 4682 audit found 100% finite/nonnegative released labels but Spearman
  0.6217465713409153 versus canonical, below the frozen 0.95 gate. `importance_released`
  is therefore not primary. Rejected alternative: ignoring the audit failure.
- **choice:** follow SPEC fallback by discovering all 6,474 raw resample files, but pause
  before downloading their 20.996 GB because measured projection exceeds one hour and
  AGENTS §8/§13 require user runtime/budget approval. Rejected alternative: starting an
  unapproved expensive data transfer.
- **fact:** GPU readiness now passes in the Codex environment: A100-SXM4-40GB,
  driver 580.105.08, locked torch 2.7.0+cu126, BF16 kernel, pinned model resolution,
  writable persistence and production doctor. Stage 6 measured fit remains pending;
  no Stage 5–7 workload ran. This supersedes only the access observation, not the
  explicit user pause on GPU-dependent experimental stages.

## Stage 5 development

- **choice:** Stage 5 generation uses immutable completion work units, the frozen
  `sample_seed` derivation, localhost vLLM requests, and benchmark outputs in a
  separate namespace. The missing `configs/pilot_targets_4682.yaml` was restored
  as an explicit manifest for the already frozen paper-display indices and adjacent
  fork mapping; no sentence numbering was inferred from new data. Rejected
  alternative: embedding benchmark outputs in production sample identities.
- **fact (Stage 5 fork audit):** the frozen Problem 4682 mapping resolves target/
  with associations `(6,6,7)`, `(12,12,13)`, `(13,13,14)`, `(14,14,15)` and
  `(46,46,47)`. The canonical union is eight forks `[6,7,12,13,14,15,46,47]`,
  with forks 13 and 14 each referenced by two adjacent target associations.
  Adaptive eligibility is limited to the five without forks `[6,12,13,14,46]`,
  because the specification tests dissimilar support for each target's without
  distribution. Rejected alternative: multiplying five targets by two.
- **choice (audit fix):** physical completion identity is now
  `(trajectory_id, fork_idx, sample_idx, seed, prompt_hash, config_hash, input_hash)`;
  target sentence and condition remain association metadata. The production planner
  emits one job per unique fork/sample and exposes all target associations for label
  assembly, including shared adaptive extensions (implementation commits `feca1dd`,
  `5baa359`). Rejected alternative: retaining
  target/condition in the rollout key, which would duplicate shared forks.
- **fact (Stage 5 historical support audit):** surviving Stage 4 support units retain
  only aggregate counts (`sample_count`, `similar_count`, `dissimilar_count`, and
  `no_answer_count`), not generated replacement text or per-sample cosine values.
  The deleted raw fallback corpus is therefore insufficient to reconstruct the
  requested per-fork similarity distributions or independently rehash selected
  samples. The five adaptive-eligible rows each show 0/100, but Stage 4 does not
  record enough DeepSeek revision, prompt, stopping, or extraction provenance to
  calibrate that rate for Stage 5. Classification: insufficient evidence; the fresh
  32-sample rule remains authoritative. Rejected alternative: treating 0/100 as a
  Stage 5 probability or generating beyond the specified 100 samples.
- **choice (Stage 5 extraction audit):** derive the first replacement sentence
  centrally with the pinned Thought Anchors `split_solution_into_chunks` semantics
  (optional `<think>` delimiters, paragraph and `.?!`-with-whitespace boundaries,
  short-chunk merging, and no unterminated final fragment). Backend-provided
  replacement fields remain diagnostics and cannot supply scientific text. Pinned
  MiniLM scores are computed from the canonical extracted text and each target
  association is scored separately for shared forks. Failed extraction or scoring
  is invalid support, never an implicit similar/dissimilar sample. Rejected
  alternative: trusting optional backend fields or converting failed output to a
  binary similarity decision.
# Stage 5 production admission controls (2026-09-14)

- **fact** — At parent `a6bf11f124297124e14e583aa6ecde15336f7c9e`, the CLI
  created a new run on every `--resume`, validated token counts only in benchmark
  mode, and lacked periodic generation checkpoints and a budget stop. Existing
  95-test coverage did not establish these long-run properties.
- **choice** — The implementation in the commit introducing this entry uses an
  explicit prepared run ID, existing manifest compatibility checks, a base-only
  256-job guard, exact token-ID comparison for each fork, the existing Runtime
  heartbeat and W&B fallback, and 15-minute incremental primary checkpoints.
  Rejected alternative: launching the old CLI based solely on passing unit tests.
- **choice** — Persist the raw generated response as a separate immutable unit
  before derived extraction/scoring. A scoring interruption reuses this raw
  response; valid completed samples remain unchanged. Nonfinite similarity is a
  failure, never support. No scientific definition or generation parameter changed.
- **fact** — User authorized base samples 0–31 only, with a stop if the projection
  after 32 completions exceeds 10 hours or $20. Adaptive samples remain unapproved.
  More than 5% truncation also stops for diagnosis. Existing benchmark measurements
  are reused because prompt/model/sampling semantics are unchanged.
- **fact** — Draft PR #2 is stacked on codex/mvp; PR #1 remains unchanged.

## A6000 Stage 5 batching (2026-09-15; commit introducing this entry, parent fce899f)

- **fact** — Actual GPU is RTX A6000, 49,140 MiB, driver 580.105.08, torch
  2.7.0+cu126. Doctor passes; pinned vLLM image is installed. Restored Stage 3
  Parquet checksums exist; no restored Stage 5 completion payload exists.
  Rejected alternative: counting historical progress metadata as completed work.
- **choice** — User-authorized runtime amendment
  `artifacts/amendments/20260915T102700Z.md` supersedes SPEC's A6000 concurrency-one
  guidance. Benchmark B2 then memory-admitted B3 with unchanged prompts, seeds,
  sampling, tokenizer, model and similarity semantics. Runtime has a separate
  resolved hash, leaving canonical rollout IDs invariant. New production run ID.
  Rejected alternative: changing scientific config to encode scheduler settings.
- **fact** — B2 startup has 78,768 KV tokens, 27.61 GiB model load, 14.42 GiB KV;
  dimensions match 48 layers × 8 KV heads × 128 × K/V × BF16 = 196,608 B/token.
  B3 is eligible; B4 is below the 15% max-context margin and is not tested.
- **choice** — Independent stream clocks replace global counter deltas for request
  timings; wall-clock aggregate throughput drives comparison, ETA and cost at
  user-supplied $1.09/h. Model-reported CUDA compatibility 13.0 is distinct from
  torch's CUDA 12.6 build. Reject summing overlapping request durations for cost.
- **choice** — Raw UnitStore commits happen in worker threads before their slots
  refill; MiniLM CPU scoring occurs only after all 256 and vLLM shutdown. Queue
  tests cover response/seed identities, duplication, corruption, no rewrites,
  continuous refill and SIGKILL recovery. Never share mutable scoring state.
- **fact** — Inspection found whole-continuation answer normalization and an
  incorrect unequal-support smoothing helper in historical code. New fresh-pilot
  label construction follows literal SPEC §3.6 and pinned upstream answer parsing;
  tests use unequal category counts and hand-calculated probabilities. Historical
  artifacts are not changed and need a separate audit before interpretation.
  Rejected alternative: silently reusing faulty helpers or silently replacing old
  results. No scientific formula is changed.

## 2026-09-15 remediation of the PR red-team findings (commits `69c05c9`..`HEAD`)

- **fact:** the pre-fix Stage 4 pipeline did not implement SPEC §3.6. The released
  `chunk_removed` field is the original sentence, so the "without" filter in
  `importance.py` kept all 100 samples; support units embedded that same text against
  itself; the smoothed KL used per-distribution category counts; answer normalization
  removed only closing braces; `importance_legacy` was a copy of canonical. All Stage 4
  and Stage 5 outputs produced before the fix are marked `PRE_FIX_NON_CANONICAL.json`
  (index `artifacts/PRE_FIX_NON_CANONICAL_INDEX.json`) and retained as evidence.
- **fact:** the Stage 5 production run `20260915T111421Z-ab6207e1fa-98852b69` was
  stopped with SIGTERM (one raw completion, failure record written) and the vLLM
  container was stopped before any repair; GPU memory returned to 1 MiB.
- **choice:** canonical importance now uses only fork-i samples whose replacement has
  pinned-MiniLM cosine similarity strictly below 0.8 to the original sentence, the
  fork-(i+1) distribution unpooled, the union of normalized answer categories including
  `<NO_ANSWER>` with one K for both distributions, and alpha 1e-9 (0.5/1.0 recorded
  as sensitivities). Rejected alternative: keeping any similar replacement.
- **choice:** `importance_legacy` reproduces the pinned repository arithmetic exactly
  (`smoothing = 1.0` module default, similar-chunk pooling into the "with" side,
  missing answers dropped, normalization applied three times along the call path),
  verified by executing the pinned functions as a test oracle. Rejected alternative:
  a "legacy-like" approximation.
- **choice:** answer categories use the pinned `normalize_latex` (MIT, ported with
  notice). Rejected alternative: the bespoke normalizer.
- **choice:** eligibility masks are applied in the primary path: first token, final
  answer sentence (last sentence containing `\boxed{`, and everything after it),
  missing adjacent fork, convergence (five consecutive forks with accuracy in
  [0, 0.01] or [0.99, 1]; K=4 recorded as `eligible_k4`), no dissimilar samples, and
  non-finite importance. Ineligible rows carry no primary label and a reason; the
  validator enforces one uniform `label_source`. Rejected alternative: unmasked rows.
- **choice:** raw resamples for all 40 trajectories are pinned in
  `configs/locks/raw_resample_*` and fetched as immutable units in a provenance run
  (`20260915T113709Z-734a9a5a41-5c37eb5c`, 6,474 files, 20,996,423,633 bytes, source
  manifest SHA-256 `43b17324cd2c9cd41b434527e1f0b7a97831fd3f5d1157ad98343e81f1ea3671`).
  The audit and build reference that run by hash so label fixes never force a
  re-download. Rejected alternative: hard-coded paths and environment-variable run IDs.
- **fact (audit):** run `20260915T120136Z-b1a2dd9ab0-3b77595c`: Problem 4682 correct,
  144 rows with an adjacent fork, 126 compared (18 have no dissimilar replacement),
  released finite/nonnegative fraction 1.0, Spearman released vs canonical
  **0.857966**, released vs legacy **0.744655**, canonical vs legacy 0.680600. The
  frozen >= 0.95 gate still fails; `importance_canonical` remains primary
  (`label_source = canonical_fallback`). The pre-fix value 0.6217 was an artifact of
  the missing filter.
- **fact (build):** 6,474 rows, **4,019 eligible** (K=4: 3,902); exclusions
  convergence 1,807, no dissimilar samples 608, missing adjacent fork 21, final answer
  19. Train-only thresholds q80 0.816651, **q90 1.354981**, q95 1.827347; 344 anchors
  (train 265, validation 54, test 25). Mean dissimilar count 52.7 (median 57); 4,973 of
  6,434 scored sentences have at least ten dissimilar replacements. Pre-fix values
  (6,434 "eligible", q90 1.068565) are superseded. Spearman released vs canonical over
  all eligible rows is 0.8914.
- **choice (Stage 5):** rollout input hashes are keyed by location-independent names;
  projections are measured from a named benchmark run at the user-supplied
  `PREANCHOR_HOURLY_RATE`; every request sends `add_special_tokens=false` and asserts
  the server's prompt token count per unit; read timeouts are bounded; the cost stop
  uses wall-clock session time and the truncation stop needs at least 20 rows; W&B
  snapshots are de-duplicated. The live vLLM tokenize endpoint returned identical IDs
  with the flag absent, true or false, so no generated prompt was affected; the flag
  removes a dependence on server defaults. No scientific parameter changed.
- **fact:** three runs created today failed before doing work and are retained:
  `20260915T113457Z` and `20260915T113541Z` (root-owned hub cache), `20260915T120105Z`
  (MiniLM download into the same cache). Transit downloads now use the user-owned
  `cache/huggingface/datasets-hub` (also via `HF_HUB_CACHE` for the audit/build).
- **choice (Stage 5, H100 session 2026-09-15):** the Stage 5 runner no longer caps
  in-flight requests at three. `batching.MAX_CONCURRENCY = 16` is a profile ceiling; the
  admitted concurrency is still derived from the measured GPU KV capacity with the 15%
  worst-case-context margin (`required_capacity`). `start_vllm.sh` reads only
  `gpu_memory_utilization` from the selected `configs/lambda/*.yaml` profile (H100 PCIe
  0.90, A6000 0.92) and keeps `max_model_len` 20480; the profile's 32768 is never used for
  Stage 5. Admission accepts the measured cards in `a6000.HARDWARE` (RTX A6000, H100 PCIe)
  and records their vendor bandwidth/BF16 figures for roofline context only. Selection
  generalizes the B2/B3 rule to an ascending chain: a larger safe concurrency is chosen
  only when it improves aggregate throughput by at least 10% over the next-smaller safe
  one. Rejected alternative: SPEC §7.5's fixed "at most four in flight on H100", which is
  an execution-only default, not a scientific setting; no sampling, prompt, seed, model,
  dtype or context parameter changed.

## Stage 6 activation-extraction smoke (2026-09-15, branch `codex/stage6`, base `abf7ea5`)

- **fact:** SPEC Stage 6 is the 16-boundary smoke; full extraction is Stage 7 and remains
  unauthorised. Stage 5 run `20260915T144926Z-bb9ba695de-9fca4c21` is finalised but its
  secondary backup is unverified (`backup.verified: false`); that gates Stage 7, not Stage 6.
- **fact (measured, H100 PCIe, disposable probes):** bf16 load 49 s, 28.2 GB VRAM; the
  longest trace (`3935:incorrect`, 13,087 tokens) replays in 3.0–3.7 s at 30.0 GB peak;
  repeat replays are bitwise identical; same-shape future-token mutation changes no boundary
  site; truncated-prefix replay differs for prefixes of 70 and 86 tokens (max abs 4.5 and 2.0
  at layer 48) and is bitwise identical for the 14 prefixes of 112+ tokens.
- **choice (user-authorised amendment `docs/amendments/20260915T170000Z.md`, copy under
  `$PREANCHOR_ROOT/artifacts/amendments/`):** the primary invariance gate is the same-shape
  future-token mutation at the frozen tolerance; truncated-prefix replay is a recorded
  diagnostic. No configuration value or hash changed. AGENTS §5.1 still words the gate as
  prefix-only equality; AGENTS is amended only on explicit instruction, so the wording
  conflict is recorded here for the user. Rejected alternative: loosening the tolerance.
- **choice (user decision):** activations are stored for all 6,474 Stage 4 rows; the index
  carries `eligible`, `eligible_k4`, `exclusion_reason`, `label_source` and the labels hash
  so the 4,019 eligible rows are a selection, not a re-extraction. SPEC §5's shape
  `[number_of_sentences, 49, 5120]` permits this; no schema invariant conflicts.
- **fact (tokenisation):** 2,844 of 6,474 sentences (1,930 eligible) begin after a space
  whose token spans the space and the first word; per SPEC §3.8 that token is target, so the
  boundary is the last token of the previous sentence, while the label-generating prefix
  `full_cot[:char_start]` tokenises to the same prefix plus one `" "` token (2,816 rows;
  Stage 5 forks 6, 7, 12, 14, 47). Recorded in `docs/alternative_explanations.md`; the
  frozen boundary definition is unchanged.
- **choice:** units are one trajectory each, keyed by `ids.cache_key` over the scientific
  config hash, model and tokenizer revisions, trace hash, token-id hash, boundary-mapping
  hash, code commit and the Stage 3/4 input hashes; unit contents are deterministic (timings
  live in attempt records) so a regenerated unit is byte-identical. Shards are named
  `<problem>_<kind>.safetensors` (the Stage 5 convention) because ids contain a colon.
  Per-attempt summaries are replaceable current views with immutable attempt copies.
- **fact:** the worktree's vendor submodules were uninitialised (`provenance()` would fail);
  `git submodule update --init` restored the pinned commits without changing the lock.
- **fact (Stage 6 smoke, commit `f9dd294`/`1907c1c`):** runs `20260915T164208Z-29d5f2acb0-7a4c0acd`
  (16 boundaries), `20260915T165230Z-8b49a551f6-1dfccd21` (SIGTERM drill, 32 rows),
  `20260915T165641Z-8b49a551f6-c8fcfdfe` (SIGKILL drill, 32 rows) and
  `20260915T170028Z-8b49a551f6-36843658` (389-row envelope) all complete and validate. The
  same-shape future-token mutation gate is exactly zero at every site for all 469 boundaries;
  truncated-prefix replay differs only for prefixes below 108 tokens (max abs 12.0), which is
  recorded as the sequence-length diagnostic. Regenerated units are byte-identical to the
  originals. Full evidence: `docs/stage6_validation.md`.
- **fact:** the first drill run `20260915T164439Z-29d5f2acb0-586ddfe5` failed because the
  offset check rejected coinciding byte-level spans of a multi-byte character; fixed with a
  regression test in `f9dd294`; the failed run is retained. Rejected alternative: deleting it.
- **fact:** Stage 7 admission: measured fit PASS, projection 410 s / $0.28 for 40 units and
  3.31 GB; BLOCKED on the Stage 5 secondary-backup receipt and on explicit authorisation.
- **choice (portable backup, 2026-09-15):** canonical state is protected as exact bytes in
  a content-addressed store (`artifacts/portable/objects/sha256/...`) with immutable,
  self-hashed bundle manifests and append-only receipts; the MacBook mirrors it under
  `~/preanchor-backups/portable/` and counts a bundle as protected only after its own
  rehash writes a `pull_verification` receipt. Scientific identity (config hashes,
  revisions, Stage 3/4 hashes, run IDs, plan and sorted rollout-ID-set hash, frozen
  generation settings) is recorded separately from physical origin (root, mount, host,
  GPU, region), which `restore verify` prints but never compares. Rejected alternatives:
  whole-run rsync snapshots (no deduplication, no per-object verification), regenerating
  Stage 3/4 on the new filesystem (byte-different `trajectories.parquet` and MiniLM
  fingerprints changed every rollout ID in the 2026-09-15 reconstruction), and requiring
  the old filesystem UUID.
- **choice (path portability, 2026-09-15):** Stage 2/4 `local_path` validation accepts a
  recorded absolute path that names the same `<run_id>/<run-relative path>`
  (`io.same_logical_path`); unit key, run ID and upstream content hashes stay strict, and
  exactly equal legacy paths still validate. Existing Stage 3 tables are not rewritten:
  the embedded absolute `source_path` is historical provenance because its bytes are part
  of every Stage 5 rollout ID. Recommendation for the next deliberate schema/identity
  boundary (a new Stage 3 import that is allowed to change hashes): drop `source_path`
  from `trajectories.parquet` in favour of the Stage 2 unit key and upstream hash, and
  record `local_path` as run-relative, so no physical location participates in
  scientific identity. Rejected alternative: silently migrating the current canonical
  tables, which would invalidate the finalized production run.
- **fact (portable backup, 2026-09-15):** first independently verified portable canonical
  MacBook secondary copy of the finalized H100 Stage 5 lineage (production run
  `20260915T144926Z-bb9ba695de-9fca4c21`): bundle `20260915T163333Z-d565605b95187a71851608d5dd1f31ad94eaffe3ae29674084da1462e83132a2`, kind `canonical`, highest protected stage 5, 4,282 unique
  objects, 33,704,252 bytes rehashed on the MacBook, 0 missing, 0 corrupt, outcome
  `secondary_copy_verified`, receipt `receipts/20260915T163431Z-pull_verification-d565605b9518-62c783c6.json` under `~/preanchor-backups/portable/` and
  copied back to `artifacts/portable/receipts/`. This validates the content-addressed
  incremental design end-to-end against the real finalized state. A bundle prepared on
  Lambda/NFS (`prepare` receipt, `secondary_verified: false`) and an independently
  verified secondary copy (`pull_verification` receipt) are distinct states; only the
  latter satisfies the backup gate. The MacBook path and filesystem identity are
  provenance only and never enter scientific identity. Rejected alternative: treating
  the NFS bundle or the run manifest's `backup` field as verification.

## Stage 6 operational closure (2026-09-15, commits `afd3714`..`128e1d3` on `codex/stage6`)

- **fact:** Stage 6 is finalized at `542ee91`; Stage 7 has not started. The portable backup
  implementation from `codex/mvp` was merged into `codex/stage6` (`afd3714`) so Stage 6
  classification, tests and the real round trip live on one branch.
- **choice:** Stage 6 canonical state is every complete Stage 6 run whose recorded inputs
  are the current Stage 3/4 tables (`discover_stage6`), with activation units under
  `activations/units` validated by `UnitStore`, and `STAGE_DEPENDENCIES[6] = (3, 4, 5)`
  because the MVP continuation after Stage 6 needs the fresh pilot. Failed or prepared-only
  runs and quarantine directories stay historical. Rejected alternative: promoting only the
  SPEC smoke run and dropping the drill and envelope evidence, or bundling all runs by
  directory name.
- **fact:** bundle `20260915T180804Z-f4cd4b0b…` (152 new objects, 216 MB incremental over the
  verified Stage 5 store) restored into an empty local-disk root and passed
  `restore verify --for-stage 6`; the MacBook secondary verification is pending and is not
  claimed. Physical origin/destination are recorded, never compared.
- **fact:** the MacBook wrapper's remote checkout defaults to `~/preanchor-repo`
  (`codex/mvp`, no Stage 6 support); the Stage 6 pull must set
  `PREANCHOR_REMOTE_REPO=/home/ubuntu/preanchor-stage6` (or the checkout containing
  `128e1d3`). Rejected alternative: modifying `~/preanchor-repo` from this session.
- **fact (2026-09-15T18:19:39Z):** the MacBook independently verified Stage 6 bundle
  `20260915T180804Z-f4cd4b0b8f03569ba33b87502010f3d015214a3157469b56d947faf1d73e6d7d`: 4,434
  objects, 250,021,342 bytes, 0 missing, 0 corrupt, outcome `secondary_copy_verified`,
  receipt `artifacts/portable/receipts/20260915T181939Z-pull_verification-f4cd4b0b8f03-d185f734.json` copied
  back to Lambda with a matching sidecar. The "MacBook secondary verification pending" state
  is resolved and the Stage 6 operational backup gate is satisfied. The append-only receipt
  is the evidence; no canonical run manifest was rewritten and no replacement bundle was
  prepared while Stage 7 runs in another session.

## Stage 7 full activation and feature extraction (2026-09-15, branch `codex/stage7`)

- **fact:** Stage 7 admission at launch commit `3c2d4ec`: the Stage 5 MacBook secondary
  copy (bundle `20260915T163333Z-d565605b…`, receipt `20260915T163431Z-pull_verification-…`)
  re-verified with the current `preanchor backup verify` (4,282 objects, 0 missing, 0
  corrupt); production doctor ok; 250 CPU tests and the GPU invariance test pass; projection
  regenerated at the launch commit (215 s, $0.15, 3.31 GB); 1.94 PB free on NFS. Rejected
  alternative: treating the run manifests' `backup.verified: false` as the backup state.
- **choice:** Stage 7 work lives on `codex/stage7`, branched from the `codex/stage6` merge
  commit `afd3714`, because a sibling session held `codex/stage6` checked out with
  uncommitted backup changes and per-stage branches (`codex/stage5`, `codex/stage6`) are the
  established convention. No PR was merged.
- **fact (production, run `20260915T182306Z-3ab408d80a-7452f1c2`):** 40 units, 6,474 rows,
  187,579 tokens, 29.1 s of forward time at 6,436 tok/s, peak 29,973 MiB, 322 s extraction
  wall, 3,314,695,040 shard bytes; `validate-activations`, `build-dataset` (61 s) and
  `validate-features` (256 rows recomputed exactly) pass; no failure, resume or quarantine.
  The full `3935:incorrect` shard is byte-identical to the Stage 6 envelope run from
  commit `f9dd294`.
- **choice (observables):** next-token statistics apply the pinned `model.norm.weight` and
  `lm_head.weight`, loaded alone from the snapshot shards and upcast to float32, to the
  stored float16 `layer=48` states (`features.statistics_dtype: float32`). The Stage 6
  diagnostic `logit_features.parquet` used the bfloat16 head and differs by up to 0.03–0.05
  on the smoke rows; it remains a Stage 6 diagnostic. `previous_sentence_tokens` is the
  token span `[target_token_start, target_token_end)` of `S_{i−1}` (zero for the first
  sentence). `generated_token_count` counts full-trace tokens after the stored prompt using
  the trace offsets (no substring re-tokenisation; the prompt decode round-trips exactly in
  40/40). `diagnostic__relative_completed_trace_position` is `sentence_idx / n_sentences`.
  Rejected alternatives: computing statistics in the extraction process with the bf16 head;
  defining the previous-sentence length as the distance between sentence starts (it counts
  inter-sentence whitespace).
- **choice (dataset layout):** the feature table carries identity, `split`, the 12 primary
  observables, the two diagnostics, boundary fields, the activation pointer
  (`activation_run_id`, `shard_path`, `row_idx`) and the Stage 4 target columns; support
  counts, convergence and final-answer indices stay in `labels.parquet`. Published path is
  `data/processed/stage7/features.parquet` (the repository's per-stage convention; SPEC
  writes `data/processed/features.parquet`). Only the `mvp` configuration publishes; a
  canonical table with different content is never overwritten; a stale smoke publication is
  relocated to `data/processed/stage7/superseded/<run_id>/`, never deleted.
- **fact:** the first smoke run at `1cfa152` published its 16-row table to the canonical
  path because publication was unconditional; fixed in `3c2d4ec` (fail-closed), the smoke
  was rerun at `3c2d4ec` and the production build relocated the stale files
  (`superseded_publication` in `feature_audit.json`). The run and its files are retained.
- **fact:** the `9bad4b36…960ff` / `14e279c5…fabd7` hashes recorded for `configs/mvp.yaml`
  in `docs/stage6_plan.md` and `docs/stage6_validation.md` belong to `configs/pilot.yaml`;
  `configs/mvp.yaml` has resolved to `c8414238…6ef4f` (scientific `82e38884…1a948`) since
  Stage 0 (`docs/stage0_validation.md`). No configuration changed; the Stage 6 documents
  are left as written.

## Stage 7 operational closure (2026-09-15, commits `480031e`, `56aea1b` on `codex/stage7`)

- **choice:** the canonical Stage 7 run is discovered as the complete extraction whose
  `build-dataset` output hash equals the published `data/processed/stage7/features.parquet`
  on the current Stage 3/4 tables (`discover_stage7`); any other complete Stage 7 run is
  historical, two publishers block. `STAGE_DEPENDENCIES[7] = (3, 4, 5, 6)`: Stage 3/4 are
  its inputs, Stage 5 the MVP continuation dependency, Stage 6 its admission evidence.
  Rejected alternatives: bundling every run with a `stages.stage7` entry, or dropping
  Stage 6 from the closure (the launch projection lives in a Stage 6-state smoke run).
- **choice:** protected Stage 7 bytes are the run's `activations/`, `features/` and
  `stages/stage7/` trees, its manifests and config, and the published table with schema,
  audit and sidecars; `data/processed/stage7/superseded/` is historical evidence.
  `restore verify --for-stage 7` reuses the project's own validators
  (`validate_run`, `validate_features(sample=0, write_record=False)`) instead of a second
  implementation.
- **fact:** the first scratch verification rewrote the restored `feature_validation.json`
  because the verifier called the validator with its record writer on; fixed in `56aea1b`
  (read-only verification, asserted by the test). The first bundle
  `20260915T185656Z-038fb6cf…` (357 new objects, 3,118,456,529 B) and the final bundle
  `20260915T191239Z-161f21fc…` (producer `56aea1b` clean, 0 new objects) have identical
  content; both scratch restores are retained.
- **fact:** the final bundle restored into an empty local-disk root and passed
  `restore verify --for-stage 7` (140 checks, 0 failed); an independent SHA-256 comparison of
  289 files / 3.32 GB found no mismatch. The MacBook secondary verification is pending and
  is not claimed. Physical origin/destination are recorded, never compared.

## Stage 8 implementation (2026-09-15, branch `codex/stage8`)

- **choice:** Stage 8 is its own run (`fit` creates it) whose `input_hashes` are the Stage 7
  feature table, activation index, labels hash and `activation_run_id`. Rejected: appending
  to the Stage 7 run, which `check_resume` refuses because the code commit differs.
- **choice (folds):** `StratifiedGroupKFold` shuffled with
  `derived_seed(seed_root, "cv_folds")` rather than the raw `seed_root`, matching how every
  other seed is namespaced (`ids.derived_seed`); seeds and validation problems are stored
  per fold in the manifest. Rejected: `random_state=seed_root` directly.
- **choice (baseline inputs):** SPEC §3.10 names map to the frozen §3.9 columns as
  position = `sentence_idx, generated_token_count, prefix_context_fraction`; context =
  `prefix_token_count, log1p_prefix_token_count, previous_sentence_tokens`; uncertainty =
  the six log-probability/entropy/margin/top-k columns; combined = all twelve. No column
  is added or dropped.
- **choice (selection):** every family selects its own hyperparameter (and site) from
  train CV; the frozen binary and Ridge selections are those of
  `activation_plus_observables` (`evaluation.primary_model`). Ridge selection uses Spearman
  with RMSE tie-break on the `log1p` scale, the scale the probe is fitted on. Remaining ties
  break to the smaller hyperparameter, then the lower site. The residualised control is
  fitted only at the selected binary site.
- **choice (integrity):** units contain no timing or attempt identifiers so regenerated
  units are byte-identical; `select` refuses to run twice on one run (a new selection needs
  a new `run_id`); a smoke configuration reads the feature table inside its activation run
  while `mvp` reads the canonical publication.
- **fact (smoke, run `20260915T191139Z-30a5e56f6f-5c6fdbe6`):** two-site real-data smoke
  fits, selects and validates; interruption, resume, corruption/quarantine and deterministic
  identity are covered by `tests/integration/test_probe_fit.py` on a synthetic Stage 7 run.
- **interpretation:** smoke CV numbers are training-fold diagnostics of two sites and carry
  no evidence about probe performance; no validation or test metric exists.
- **fact (production):** run `20260915T191905Z-a2bcc389c7-fc030488` was launched with an
  untracked log file inside the worktree, so its identity recorded a dirty tree and `select`
  refused to continue once the log grew; it is superseded and retained. The canonical run
  `20260915T193326Z-57bcb41c84-7ebd0db6` (clean tree) reproduced the same
  `cv_results.parquet` bytes (`68ea55b0…`). Logs now live outside the repository.
- **fact (backup gate):** no stage-7 `pull_verification` receipt existed at launch; the user
  authorised production explicitly. Recorded in `stages/stage8/resource_projection.json`
  and `docs/stage8_validation.md`; the Stage 7/8 receipts remain required before Stage 9.
- **interpretation:** the pending selection (site 44, C 0.1; Ridge site 28, alpha 1e4) and
  all AUCPR values in `docs/stage8_validation.md` are training-fold diagnostics, not
  evidence about held-out performance.

## Control plane v0.1 (2026-09-15, branch `agent/ops/control-plane-v1-20260915-0bd79efc`, base `f6ce263`)

- **fact:** Git worktrees do not isolate the H100, the `preanchor-vllm` container,
  `$PREANCHOR_ROOT`, the canonical publication paths or the portable store; separate
  sessions were already running from separate worktrees (`codex/stage7`, `codex/stage8`)
  against the same root.
- **choice:** ownership is a persistent per-resource directory under
  `$PREANCHOR_ROOT/control/leases/`, created with one `mkdir` (atomic on the virtiofs root)
  and ended with one `rename` into `.archive/`; `owner.json` stores only the SHA-256 of a
  random owner token, and only that token releases. Rejected: relying on the existing
  process-scoped `fcntl.flock` (dies with the process, invisible across sessions), a lock
  file with `O_EXCL` (a killed writer leaves an empty file that carries no owner
  information), or any daemon/database.
- **choice:** stale heartbeats never transfer ownership; `reclaim` needs a reason and
  either positive evidence (owner session closed, or an incomplete acquisition older than
  the stale window) or `--force` on a stale lease; a live owner process on this host
  always blocks. Rejected: inferring death from heartbeat age or from `nvidia-smi`.
- **choice:** gates sit at the real mutation points, before the expensive import or the
  first store write: `model.load.load_model`, `features.observables.load_head` (non-CPU),
  `scripts/start_vllm.sh` (after argument validation, before Docker), `backup.prepare.prepare`
  and `backup.legacy.import_legacy` (after the read-only plan and its blockers). The CLI
  auto-leases the writer for `backup prepare`/`import-legacy` through a short-lived
  `backup-writer` session so `scripts/preanchor_portable.sh` is unchanged. Rejected: an
  advisory-only design, and requiring manual lease commands for every routine backup.
- **interpretation:** the GPU-only scientific invariance test loads the model and is
  therefore gated too; a session that runs it holds `gpu:h100` exactly as extraction does.
- **choice:** `control/` is operational metadata: outside run manifests, config/unit
  hashes, run IDs and `scientific_identity_hash`, and outside portable bundles (the
  classifier does not scan it). Including receipts as operational provenance is deferred
  to v0.2 rather than changing bundle contents silently. Tests that mutate a synthetic
  store now hold the writer lease through `backup_fixtures.grant_writer_lease`; no
  scientific test was weakened.


## Control plane v0.1 independent review (2026-09-15, branch `agent/ops/control-plane-review-20260915-e1d2bc32`)

- **fact:** review of `dde76ce` from a fresh worktree reproduced the CPU suites (286 passed /
  1 skipped; scientific 5 passed / 1 skipped) and found one normal-path GPU bypass: the
  documented `audit-upstream` / `build-labels --device cuda` commands load MiniLM through
  `labels.similarity.ReplacementSimilarity` / `labels.generate.MiniLMSimilarityScorer`
  with no lease.
- **choice:** gate those two constructors on a non-CPU device with `require(["gpu:h100"])`
  before `sentence_transformers` is imported. Rejected: gating `preanchor doctor`'s 8x8
  admission kernel (telemetry) and gating the CPU default.
- **choice:** resource names may not contain `__` (the on-disk escape of `:`), so
  `gpu__h100` can no longer alias `gpu:h100`'s directory. Rejected: a different escape
  character (would change the validated on-disk layout).
- **choice:** the MacBook `restore` holds `portable-store:writer` around its rsync push
  through a short-lived `backup-restore` session and releases it on every exit path;
  read-only remote steps and the root-side `restore apply` stay unleased. Rejected: moving
  the push behind a new CLI command (a redesign of the transfer path).
- **interpretation:** the remaining limits (per-root authority, `PREANCHOR_CMD` override,
  one-time gate check, receipt-before-rename) are documented in `docs/concurrency.md`
  and do not allow two owners or an unaudited takeover.

## Stage 9 implementation and execution (2026-09-15, branch `agent/stage9/validation-freeze-20260915-46c93dfe`)

- **choice:** Stage 9 is its own run whose `input_hashes` name the Stage 8 run and its four
  selection artifacts, so `check_resume` binds `calibrate`, `freeze-selection` and
  `evaluate-test` to one code identity (`67f824a`). Rejected: appending to the Stage 8 run
  (its identity is `0d07354`).
- **choice (calibrator):** Platt (1999) sigmoid on the selected probe's raw decision score,
  maximum likelihood against the prior-corrected targets, BFGS, fitted on validation rows
  only and applied only to the selected binary probe. Rejected: sklearn's private
  `_SigmoidCalibration` (same estimator, unstable API) and isotonic regression (not the
  specified method).
- **choice (freeze hash):** the freeze is the sorted-key JSON of every selection-relevant
  identity and its SHA-256 is the freeze hash; validation metrics enter only as hashes of
  the already written files, never as inputs to any choice.
- **choice (exactly-once):** claim in an immutable manifest revision before reading test,
  byte-deterministic outputs written with the non-replacing atomic writer, receipt as commit
  point; a crashed attempt is retried under the same hash and reproduces identical bytes or
  fails. Rejected: refusing any retry after a crash (would turn an I/O failure into a burned
  protocol) and a staging-directory rename (not atomic across three files).
- **choice (bootstrap):** per-trajectory top-10 % statistics are precomputed once and
  averaged with multiplicity inside each replicate; all models share one problem draw per
  replicate; seed `derived_seed(seed_root, "bootstrap", split)`.
- **choice (backup):** the canonical Stage 8 run must fit `n_layers + 1` sites, so the
  two-site smoke fit on the same inputs stays historical; Stage 9 canonical = the frozen run
  on the canonical Stage 8 selection, blocker when two are frozen.
- **fact:** validation: 54 anchors ≥ 20 → calibration fitted (A −0.11335, B 1.88096). Test
  evaluated once: 25 anchors, primary AUCPR 0.0941 (0.036 … 0.162), gain over the combined
  observables 0.037 (−0.051 … 0.081), position-only baseline AUCPR 0.139. Residualised
  control 0.127. No parameter, threshold, layer or calibrator was changed afterwards.
- **interpretation:** exploratory held-out engineering test on four public problems; the
  probe exceeds prevalence and the combined observables in point estimate, the paired AUCPR
  interval against the combined observables includes zero, and a position-only baseline is
  competitive. No significance claim; Stage 10 controls remain prespecified and unrun.

## Stage 9 provenance audit and acceptance (2026-09-15, branch `agent/ops/stage9-audit-20260915-16702032`)

- **fact** — The Stage 9 run `20260915T201107Z-484989d2fe-19f1ab0d` was produced at clean
  `67f824a`; its immutable manifest revisions order calibrate (20:11:09–20:12:04) →
  freeze (20:12:20, `e68f1d82…`) → test claim (20:12:46, same hash, `resumed: false`) →
  test complete (20:13:32, receipt `a6ef2ba8…`); one `evaluate-test` is recorded, the
  second invocation (20:13:48) was refused before the claim. Details and evidence paths in
  `docs/stage9_audit.md`.
- **fact** — Stage 8 inputs (`selected_model.joblib af63687f…`, pending selection
  `e1161a5a…`, CV tables `d3aeae70…`/`e755638d…`), the train-only q90
  `1.3549810893213545`, the CV argmax (site 44 / C 0.1; site 28 / alpha 1e4), the
  validation-only Platt fit (680 rows, 54 positives; refit reproduces A/B; a val+test fit
  would differ) and the four test problems `{2236, 2870, 3916, 6998}` were re-derived
  independently; 810 saved metric/bootstrap values reproduce from the saved predictions
  (worst |diff| 5.6e-17).
- **choice** — Classification **ACCEPT**; the run is not rerun and its producing commit
  remains `67f824a`. Alternative rejected: re-evaluating under the newer control-plane code,
  which would be a second look at the same four test problems for no scientific gain.
- **choice** — Integration by one `--no-ff` merge of `8f4b20a` into `a0ebdc3` (`63e2e55`),
  docs conflicts resolved by appending the Stage 9 sections after the integration record.
  Alternative rejected: rebasing the Stage 9 branch onto `a0ebdc3`, which would rewrite
  `67f824a`.
- **choice** — Post-test changes on the Stage 9 branch (`9e71b1f`: backup classification
  and verification, read-only `validate_stage9(write_record=False)`, a synthetic fixture
  constant; `8f4b20a`: documentation) are classified operational; no selection-relevant
  field changed, so the result stands.
- **choice** — The Stage 9 bundle was re-prepared under the reviewed writer lease
  (`20260915T212836Z-99e76aa8…`, 0 new objects) so that `latest.json` names a stage-9
  bundle for the MacBook `--no-prepare` pull; the unleased `8605ff94…` bundle and the
  verified Stage 8 bundle `ae33090e…` are preserved. The result is primary-only until the
  MacBook `pull_verification` receipt exists.

## Stage 10 controls, sensitivities and lead time (2026-09-15, branch `agent/stage10/controls-20260915-0ede0550`, commit `c1e303c`)

- **fact:** Stage 10 ran from the finalized Stage 9 run `20260915T201107Z-484989d2fe-19f1ab0d`
  (freeze `e68f1d82…`, one receipted test evaluation) as its own run
  `20260915T211903Z-b230473011-815ca2b7`; every Stage 9, Stage 8 and input hash was
  identical before and after (`docs/stage10_validation.md`). The frozen selection (site 44,
  C 0.1, q90 = 1.3549810893213545, validation-only Platt) was read, never changed.
- **choice (base):** the worker branch starts at the Stage 9 head `9e71b1f` and merges the
  reviewed control plane `a0ebdc3` (`a8ac6ac`; the only conflicts were the order of appended
  documentation sections). Rejected: basing on the integration branch alone (no Stage 9 code)
  or on Stage 9 alone (no session/lease registry).
- **choice (refits):** a sensitivity that changes the training labels or rows (q80/q95, K=4,
  supported rows, lead one, permutations) refits the family at its Stage 8-frozen site and
  hyperparameter (`payload["models"]`), with no grid and no layer search; sensitivities that
  only restrict evaluation rows (subgroups, quintiles) re-score the frozen Stage 9
  predictions. Both sources are tagged in `metrics/sensitivities.parquet`. Rejected: rerunning
  Stage 8 selection per sensitivity (a new selection is exactly what Stage 10 must not do).
- **choice (thresholds):** q80/q90/q95 are `labels/threshold.py::train_threshold` (numpy
  linear interpolation) over eligible training importance only; the run refuses if the
  recomputed q90 differs from the freeze. Values: 0.8166508259882914, 1.3549810893213545,
  1.827347118920559 (identical to Stage 4's `labels_summary.json`).
- **choice (K=4):** the K=4 sensitivity changes eligibility only (`eligible_k4`) and keeps
  the frozen q90 threshold, isolating the convergence rule. Rejected: recomputing a quantile
  on the K=4 set, which would confound two changes.
- **choice (position quintile):** train-derived cut points on `sentence_idx` (25, 53, 86,
  129), the allowed §3.9 position quantity, used both as the diagnostic stratifier and as the
  permutation stratum. Rejected: within-trajectory rank, which needs the completed
  trajectory length (an excluded quantity).
- **choice (entropy quintile):** train-derived cut points on `next_token_entropy`.
- **choice (lead one):** for target S_i the features (site-44 state *and* the twelve
  observables) come from the stored row of S_(i-1), whose boundary is
  `target_token_start(S_(i-1)) - 1`; targets with `sentence_idx == 0` are stored as
  unavailable. The lead-zero comparator is the frozen model restricted to the same rows.
  Rejected: shifting first sentences to any other boundary; reusing lead-zero observables at
  lead one (they would leak S_(i-1)'s tokens).
- **choice (permutations):** 100 permutations of *training* labels within
  `(problem_id, trace_kind, position_quintile)`, seeds `derived_seed(seed_root,
  "permutation", k)`, single-class strata left fixed (45 of 95), refit at the frozen site
  and C, scored against the fixed validation/test labels; the observed statistic is the
  frozen model's. Rejected: permuting evaluation labels (a different null) or forcing balance
  across strata.
- **fact:** `position_quintile 5 / test` has 53 rows and no anchor: reported as undefined,
  not pooled. Every row has `sample_count = 100`, so the two replacement-support clauses
  coincide.
- **fact (tests):** the Stage 9 branch's `test_backup_stage9.py` predates the control plane
  and needed the writer lease; `test_backup_store.py` asserted that Stage 9 is unknown to
  `stage_closure` although `9e71b1f` had added it. Both tests were updated to the merged
  behaviour, neither was weakened.
- **interpretation:** see `docs/stage10_validation.md`. The primary gain over the combined
  observables (+0.037 test AUCPR, interval crossing zero) keeps its sign at q80, under K=4
  and on supported rows, vanishes at q95 and at lead one on test, and lies inside the
  stratified permutation null (84th percentile on test, 91st on validation). Nothing here
  reopens Stage 8/9 selection.
## 2026-09-18: research-post evidence audit (parent commit `f11823f`)

- **choice:** Reproduce headline AP and problem-bootstrap comparisons from saved predictions, and reproduce the existing transition models at their saved C values. Store provenance, predictions and figure inputs in `research/blog/`. Rejected: treating prior prose as authoritative or reopening frozen model selection.
- **fact:** The test's 15 early anchors are at absolute sentence indices 0–25, not in each trajectory's first eligible-position quintile; that relative definition gives seven. The post and new position figure distinguish these definitions.
- **choice:** Compare transition decoding with previous-sentence stage plus position (validation/test AP 0.7623/0.7924). Rejected: highlighting only the weaker position or registered-observables comparison. Independently seeded paired problem bootstraps are exploratory and recorded in `research/blog/evidence.json`.
- **choice:** Compare anchor lead zero and lead one on identical target rows. Rejected: plotting the unfiltered lead-zero score against the smaller lead-one population.
- **interpretation:** Anchor activation gains are not established beyond strong observables; smoothing sensitivity concerns the estimator, and missing-answer association is not a decomposition of KL. Transition decoding remains compatible with lexical preparation. No frozen scientific choice or locked evaluation was changed.
## 2026-10-09: redesign Phase 1 (parent commit `f11823f`; exploratory, no frozen change)

- **choice:** Reacquire the released raw forks (pinned revision, hash-checked against the Stage 4 raw run's source manifest) into a compact per-rollout table, and recompute labels on CPU/MPS. Rejected: reporting items 1–2 as blocked on saved aggregates alone. Owner authorised the download on 2026-10-09.
- **fact:** The recomputation reproduces every eligible `importance_canonical`, `dissimilar_count` and α-sensitivity value exactly. At α = 1e-9, 330/344 anchors draw at least half of I from answers absent from the "with" group; 309 of those are real answers and 21 are `<NO_ANSWER>`. `<NO_ANSWER>` totals 11,181 samples: 9,739 length-truncated, 5 parse failures, 1,437 other (which mixes unboxed stated answers with early stops).
- **fact:** Split-half κ for the q90 anchor flag is 0.195 (α = 1e-9), 0.445 (α = ½) and 0.417 (fork TV). These are lower bounds at N ≈ 50 per half.
- **choice:** Candidate rules were fixed in `research/redesign/phase1_analysis_plan.md` before the data were read. No test-split metric was computed. Rejected: scoring new labels or baselines on test.
- **interpretation:** These findings support the review's diagnosis that the current anchor flag is dominated by rare unseen answers, mostly real answers. No label is adopted; any adoption needs an amendment and owner approval. Report: `research/redesign/phase1_report.md`.
