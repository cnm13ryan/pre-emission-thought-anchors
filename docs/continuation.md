# Continuation record — 2026-09-14

## Objective and current state

Implement the replay-first MVP in SPEC stage order. Stage 0 and Stage 1 implementation,
validation and verified secondary backups are complete. **Stage 2 is now in progress:**
discover and selectively fetch the pinned released traces, labels and Problem 4682
correct-trajectory raw resamples. No dataset content or model weights have been downloaded
by Stage 2 yet; only remote tree metadata was read.

Repository: `/home/ubuntu/preanchor-repo`. Persistent experiment root:
`/lambda/nfs/preanchor/preanchor`. Branch: `codex/mvp`, HEAD `fbe40e1`.
The tracked tree was clean at Stage 1 handoff. At this record's creation, **untracked
Stage 2 work exists in `src/preanchor/data/`**; this continuation file is also new.
Do not describe the current working tree as clean or discard the uncommitted work.

The latest user instruction authorizes writing this file only. No other file was modified
for this continuation-record request. Subsequent implementation should resume under the
previous Stage 2 authorization after the compaction/handoff.

## Authority and enduring constraints

Read `AGENTS.md`, `IMPLEMENTATION_SPEC.md`, then `research/project-brief.md`; all three
were read completely before original implementation. Repository documents, committed
validation records and stored artifacts outrank this summary. Do not modify AGENTS.
The brief owns the research question/estimand; SPEC owns the detailed execution plan.

- Preserve the passive, offline, correlational experiment. No training/reward use.
- Frozen science is in `configs/mvp.yaml` and `configs/splits/mvp_20_v1.yaml`: exact model,
  revisions, BF16/no quantization, context, twenty problems and 12/4/4 grouped split,
  sentence eligibility/convergence, canonical KL and audit gate, training-only anchor
  thresholds, pre-emission timing and 49 sites, observable allow-list, linear/logistic
  probes, CV/selection, primary metric and all required controls. Read the files rather
  than reconstructing values from memory. Changes require the amendment record, new
  resolved-config hash/run ID and user approval before execution.
- Exact stored-trace tokenization once; overlapping target token excluded from prefix;
  boundary is target-token-start minus one. Layer 0 is embedding output, 1–48 are block
  outputs before final norm. No target/future/completed-length/correctness primary inputs.
- Canonical importance is smoothed KL(without || with) over normalized answer-category
  union including NO_ANSWER. Released labels require the Problem 4682 audit gate.
- Problem-grouped splits; fold-local fitting; validation-only calibration; no test
  metrics before frozen selection; one frozen-hash test evaluation. Four held-out
  problems remain an exploratory engineering test. Never hide negative findings.
- Follow minimal example → schema/checksum validation → smoke → kill/resume → scale.
  Unit/integration tests before commits, relevant tests after changes, scientific gates
  before MVP-scale work. No test weakening. Maintain decisions and alternative explanations.
- Persist valuable state under PREANCHOR_ROOT; never delete persistent data without
  explicit authorization. Atomic checksummed outputs, stable identities, immutable
  completed units, quarantine corruption and validate resume config/code/input hashes.
- Long jobs detached in tmux, main-loop heartbeat every 30 seconds. CPU stages hide CUDA.
  Required benchmarks/projections must precede Stage 5, Stage 7 and >1-hour jobs; no
  runtime/cost beyond authorization. Hourly rate and compute budget are still unspecified.
- Never provision/resize/terminate infrastructure, issue shutdown, expose public services,
  push/merge/publish/send messages or modify upstream without the relevant explicit approval.
  No root SUCCESS or SAFE_TO_TERMINATE until the complete MVP and backup gates hold.
- Upstreams stay pinned and untouched; MIT notices were inspected. Secrets remain in
  environment/untracked .env, never artifacts/logs/config/commits. No sub-agent delegation
  is authorized or in use.

## Completed milestones and durable evidence

- `2852b60`: Stage 0 bootstrap on main; then created `codex/mvp`.
- `ad5758b`: corrected persistent-mount guard for Codex virtiofs exposure.
- `b002449`: Stage 1 manifests, IDs, atomic I/O, runtime, tracking, CLI and CPU smoke.
- `fbe40e1`: persistent Stage 1 validation record and completed-heartbeat status correction.

Read `docs/stage0_validation.md`, `docs/stage1_validation.md`, `docs/decisions.md`,
`docs/environment_deviations.md`, `docs/lambda_runbook.md` and `docs/upstream_audit.md`.
Stage 0 ended with 26 tests passing. Stage 1 ended with **41 tests passing**, Ruff and
CPU doctor passing. Persistent SIGTERM and SIGKILL smoke preserved completed SHA-256
and nanosecond mtimes, quarantined two partial units, created only two missing units,
then completed a no-op resume. W&B sessions were offline and retained in the run.

Stage 1 run: `artifacts/runs/20260914T175938Z-fab7e8ae04-f1b5dc10` under PREANCHOR_ROOT.
It is bound to clean source commit `b002449e87210655813cfd8d78c953f95a803b1b`; do not
resume it under changed code. Its validation JSON hash is
`89e5701ecf2321687127970f63a21b797057b44de9d927d00a70341d93984c8d`.

MacBook is the user-approved secondary destination, `~/preanchor-backups/`. **MacBook
initiates SSH pulls; never SSH into or push to the MacBook.** Exclude caches and secrets.

Both secondary gates passed; receipts were matched against all persistent files:

| Stage | Snapshot under PREANCHOR_ROOT/artifacts | Files | Manifest SHA-256 |
| --- | --- | --- | --- |
| 0 | `bootstrap/stage0-ad5758b7d2f9` | 6 | `dcd5417acca481bce61e1e50fbf79d2305f3f7f12b879ffb8877a03615f9c22c` |
| 1 | `backups/stage1-20260914T175938Z-fab7e8ae04-f1b5dc10-fbe40e10` | 122 | `7dc74cfa250e5a4aded798faaf327819bff503cf7f13d68314578ffed07940f5` |

Stage 1 user receipt: verified `2026-09-14T18:10:50.343230+00:00`, destination
`/Users/cnm13ryan/preanchor-backups/stage1-20260914T175938Z-fab7e8ae04-f1b5dc10-fbe40e10`.
It and its checksum were saved alongside the snapshot as
`<snapshot-name>.secondary_receipt.json[.sha256]`. All 122 files were rehashed successfully
at the start of Stage 2. Stage 1 backup is no longer pending, even if older validation
documentation describes its state at the earlier handoff.

## Environment decisions and unresolved GPU issue

`/home/ubuntu/preanchor` is a Lambda symlink; the actual checkout path differs intentionally.
Codex exposes the designated persistent mount as virtiofs. Native-NFS-only checks were
too narrow; current guards require a dedicated writable mount beneath `/lambda/nfs` and
recognize native NFS or virtiofs, retaining checksum validation.

Host SSH `nvidia-smi` works according to the user; the earlier Codex sandbox invocation
failed. **This is not evidence of a host-driver failure. GPU-dependent stages remain
paused until host/Codex access is resolved with the user.** The current tool profile is
unrestricted filesystem/network with approval policy never; do not pass sandbox_permissions.
This permission change does not waive scientific or GPU gates. Do not repair drivers/CUDA.

Python 3.11.16 is in `/tmp/preanchor-python`, uv cache in `/tmp/preanchor-uv-cache`, and
the ignored `.venv` is installed with CPU/base/dev dependencies. Use `.venv/bin/python`
or `UV_CACHE_DIR=/tmp/preanchor-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/preanchor-python uv ...`.
GPU extra is not installed. No dependency pins changed; uv.lock SHA-256:
`1126f52ae681e23abc08981934af3fc2a43470da54f68ce021a38a044bd55778`.

## Stage 2 work, commands and findings

Created, **uncommitted and not yet validated**:

- `src/preanchor/data/__init__.py`
- `src/preanchor/data/fetch.py`: draft discovery/allowlist validation, upstream Git/LFS
  checks, source shape checks and per-file UnitStore download wrapper. Read before editing.

Successful read-only calls used pinned `HfApi.list_repo_tree` for dataset
`uzaymacar/math-rollouts`, revision `a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2`.
Observed root: README.md, .gitattributes, data/, Qwen-14B and Llama-8B directories.
The `data/` directory contains a large 183-shard combined Parquet export; **do not fetch it**.

Selected prefix:
`deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95`.
It contains correct/incorrect base directories plus forced-answer/sanity variants.
Use only `correct_base_solution` and `incorrect_base_solution`. The correct directory
listing matched the specified twenty problems. Full two-kind coverage is still to validate.

Observed `correct_base_solution/problem_4682` has 149 immediate entries including:

- `base_solution.json` (21,954 bytes)
- `chunks.json` (30,904 bytes)
- `chunks_labeled.json` (93,693 bytes)
- `problem.json` (810 bytes)
- chunk directories; `chunk_0/solutions.json` is 3,302,151 bytes.

No `selected_problems.json` was found in the inspected root or pinned source repository;
do not invent one or claim an exhaustive search. Exact primary file coverage and audit
fork count need the complete scoped discovery.

The draft plans to allowlist four small primary/metadata files per trajectory, README,
and all correct-4682 `chunk_*/solutions.json` files. Auxiliary `chunks.json`/`problem.json`
support later import/audit. This draft choice still needs recorded rationale and validation.
It excludes other models, sampling variants and combined Parquet exports. Downloads
are planned through pinned `snapshot_download` with exact singleton allow_patterns,
persistent HF cache and independently checksummed source copies.

**Latest command failed before discovery executed:** importing the new module raised
`ImportError: cannot import name 'RepoFile' from 'huggingface_hub'` with pinned Hub 0.33.4.
Do not repeat that top-level import or change dependency versions to solve it. Inspect
the installed API module; RepoFile/RepoFolder are likely available from
`huggingface_hub.hf_api`. No generated `dataset_files.txt` or `dataset_metadata.json`
exists yet; `configs/locks/` currently contains only `upstream.yaml`.

Other failed approaches not to repeat: treating sandbox GPU failure as host failure;
requiring native NFS fstype inside Codex; using default read-only uv cache under the old
sandbox; marking a completed heartbeat stale merely because it is old. These were fixed
and documented. The latest Stage 2 failure has not yet been fixed or tested.

## Exact next actions

1. Re-read authoritative disk state and `git status`; preserve the untracked draft.
   Record the verified Stage 1 receipt in decisions/validation status during normal work.
2. Fix the Hub type imports using installed pinned source. Remove unused imports and
   check other draft API assumptions; do not upgrade packages.
3. Run scoped discovery, validate both trace kinds for all twenty IDs, exact audit forks,
   source sizes and upstream hashes. Write/review `configs/locks/dataset_files.txt` and
   metadata/checksums. No wildcard or whole-repository content download.
4. Download the smallest representative file(s) with an explicit allowlist; inspect raw
   object/list schemas without performing Stage 3 import or Stage 4 label analysis.
   Fix draft shape assumptions using evidence. Measure throughput/size before deciding
   full download duration; if >1 hour, apply the required projection/authorization gate.
5. Implement/tests for Stage 2 CLI `fetch --discover`, `fetch [--resume]`, and
   `validate-download`, integrating existing manifests, run identity, main-loop heartbeat,
   signals, failure records and tracking. Existing CLI has no Stage 2 commands yet.
6. Test exact pins, coverage, forbidden paths, hashes, interrupted downloads, corruption
   quarantine and non-rewriting resume. Run unit/integration before milestone commits;
   follow applicable scientific/scale gates without claiming unimplemented tests passed.
7. Commit validated code/allowlist, create a new run bound to it, then run the small smoke
   and kill/resume on persistent storage before the full authorized Stage 2 download.
   Long jobs detached in tmux. Validate source manifest and every completed source file.
8. Record Stage 2 evidence and prepare a MacBook-pull backup (existing backup preparers
   are stage-specific and need an appropriate Stage 2 implementation). Await its verified
   receipt at the stage gate. GPU work remains paused throughout.
