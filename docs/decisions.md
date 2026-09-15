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
