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
  assembly, including shared adaptive extensions. Rejected alternative: retaining
  target/condition in the rollout key, which would duplicate shared forks.
