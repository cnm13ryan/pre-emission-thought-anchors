# Continuation record — Stage 2 handoff, 2026-09-14

## Objective and next gate

Implement the replay-first MVP in SPEC stage order. Stage 0 and Stage 1, including verified
MacBook backups, are complete. Stage 2 acquisition/validation is complete; its verified
MacBook secondary-copy receipt is the next gate. **Do not start Stage 3 until that gate
is recorded. GPU stages remain explicitly paused.**

Read `AGENTS.md`, `IMPLEMENTATION_SPEC.md`, then `research/project-brief.md`; disk artifacts,
committed validation records and manifests outrank this summary. Never modify AGENTS.
Repository `/home/ubuntu/preanchor-repo`, branch `codex/mvp`; persistent root
`/lambda/nfs/preanchor/preanchor`. `/home/ubuntu/preanchor` is a Lambda symlink.
Use `git status` and `git log` for current source state; source is committed before backup.

## Completed work and authoritative records

- Stage 0: `2852b60` bootstrap, `ad5758b` virtiofs guard correction; 26 tests.
- Stage 1: `b002449` provenance/I/O/runtime/tracking, `fbe40e1` validation/status; 41 tests.
- Stage 2 execution: clean `700d34899bb6e195d76702bed7acad93e1ad9f3c`.
  The subsequent commit containing this record fixes worker-timeout propagation, adds
  regression coverage and records final evidence. Final suite: 61 tests, Ruff pass.
- Read `docs/stage0_validation.md`, `docs/stage1_validation.md`, `docs/stage2_validation.md`,
  `docs/decisions.md`, `docs/upstream_audit.md`, `docs/environment_deviations.md`.
- Stage 2 adds `data/fetch.py`, `data/stage.py`, exact `configs/locks/dataset_*`, fetch and
  validate-download CLI, CPU scope/hash/resume tests, `scripts/stage2_smoke.py` and
  `scripts/prepare_stage2_backup.py`. AGENTS, frozen science, uv.lock and submodules unchanged.

Stage 2 persistent runs under PREANCHOR_ROOT/artifacts/runs:

- Smoke: `20260914T182859Z-1c085746e3-6c451e8c` — SIGTERM/SIGKILL, two partial quarantines,
  completed hashes/mtimes unchanged, five verified files, no-op resume.
- Full: `20260914T183017Z-1c085746e3-08bc1e7c` — all 306 files / 339,390,844 bytes,
  177.497 seconds, 169,984,000-byte peak RAM; no-op resume creates 0/skips 306 and preserves
  all 1,836 completed-unit files. GPU unused/unmeasured. Raw source manifest SHA-256:
  `ff5c1d6d043db2d1f8a01b2c5152bb8139e17b8f1bd5f11cb360821baee65663`.
- Minimal raw inspection and uncached benchmark: `artifacts/bootstrap/stage2-minimal/`.
- Runs retain original code identity. Do not resume them under changed code. Stage 3
  needs new provenance and explicit Stage 2 input hashes, not a forced identity update.

Commands executed: pinned tree discovery; singleton snapshot downloads; unit/integration
and Stage 2 scientific tests; CPU doctor; detached tmux smoke and full fetch; complete
source/hash validation and no-op resume. Full evidence hashes and exact commands are in
`docs/stage2_validation.md` and the run manifests/attempt records.

## Data findings and failed assumptions

Exact selected prefix: `deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95`.
Forty each of base_solution.json, chunks_labeled.json, chunks.json and problem.json;
README; all 145 correct-4682 chunk_0..144/solutions.json forks. Other models, variants
and combined Parquet exports excluded. Raw bytes were not transformed.

Hub 0.33.4 types are in `huggingface_hub.hf_api`, not the package root. `chunks.json` is
an object containing string chunks; labeled chunks are a list of objects. These draft
assumptions failed in the minimal example and were fixed/tested; do not upgrade libraries.
A worker TimeoutError must propagate instead of becoming an endless heartbeat poll.

**Correction to the original handoff:** `vendor/thought-anchors/selected_problems.json`
exists, has 106 records, and its first twenty IDs exactly match the frozen selection.
It was inspected; do not repeat the earlier assertion of absence from the source checkout.

## Durable backups

MacBook is the verified secondary destination, `~/preanchor-backups/`; **Mac initiates
SSH pulls. Never SSH into or push to it.** Exclude model/HF caches and secrets.

Stage 0: six files, receipt verified 2026-09-14T17:37:27.331293+00:00, snapshot
`artifacts/bootstrap/stage0-ad5758b7d2f9`, manifest
`dcd5417acca481bce61e1e50fbf79d2305f3f7f12b879ffb8877a03615f9c22c`.
Stage 1: 122 files, receipt verified 2026-09-14T18:10:50.343230+00:00, snapshot
`artifacts/backups/stage1-20260914T175938Z-fab7e8ae04-f1b5dc10-fbe40e10`, manifest
`7dc74cfa250e5a4aded798faaf327819bff503cf7f13d68314578ffed07940f5`.
Checksummed user receipts are beside their persistent snapshots.
Stage 2 snapshot is prepared by `scripts/prepare_stage2_backup.py FULL_RUN SMOKE_RUN`
after committing this record; inspect `artifacts/backups/stage2-*` for exact identifier.
A prepared snapshot is not secondary verification. Wait for the Mac receipt.

## Invariants and unresolved issues

- Brief owns research question/estimand; SPEC owns detailed execution and stages.
- Passive offline correlational probe, never training/reward. Frozen config choices need
  amendment, new config hash/run and user approval before execution. Read actual configs.
- Exact full-trace tokenization once; boundary target_token_start−1, overlapping token
  excluded; layer0 embedding,1..48 block outputs pre-final-norm. No future/target features.
- Canonical smoothed KL(without || with); released labels require 4682 audit. No labels
  or probes have been scientifically validated yet. No test metrics inspected.
- Problem-grouped splits, fold-local fitting/thresholds, validation-only calibration,
  freeze before one test evaluation, four held-out problems exploratory. Nulls retained.
- Minimal example → schema/checksum → smoke → kill/resume → scale. Unit/integration before
  commits and relevant scientific gates before scale; later scientific tests unimplemented.
- All valuable state under persistent root. Atomic hashes, immutable valid units, quarantine
  partial/corrupt work, strict code/config/input resume identity. No persistent deletion.
- Long jobs tmux; heartbeat main loop every30s; W&B offline records backed up. No secrets
  in outputs, no model caches in backups, no raw activation uploads.
- Host SSH nvidia-smi works per user; earlier Codex sandbox check failed. This is tooling
  context, not a host-driver failure. Stop before GPU checks/stages to agree execution.
  Permission-profile changes do not waive this gate. Never repair drivers or CUDA.
- Manual infrastructure only; no create/resize/terminate/shutdown; no push/merge/publish
  or messages without user approval; no sub-agent delegation requested.
- Benchmarks/projections before Stages5/7 and >1h jobs; hourly rate/runtime/budget remain
  unspecified. Stage2 conservative projection was19.6min; actual fetch was3min.
- No run-root SUCCESS or SAFE_TO_TERMINATE before complete MVP/backup gates.
- Python3.11.16 tools in /tmp/preanchor-python, uv cache /tmp/preanchor-uv-cache; ignored
  .venv has CPU/base/dev only. GPU extra not installed. uv.lock hash unchanged:
  `1126f52ae681e23abc08981934af3fc2a43470da54f68ce021a38a044bd55778`.

## Exact next actions

1. Verify current Git/artifact state and Stage2 snapshot identifier/checksums.
2. Obtain user-provided MacBook pull verification receipt; rehash primary snapshot,
   match count/manifest digest, save checksummed receipt and update gate records.
3. Then start SPEC Stage3 minimally: inspect upstream splitter/answer helpers and raw
   schemas; retain MIT notices for ports; implement import/segmentation/exact sequential
   alignment, pinned tokenization and pilot target text mapping under existing gates.
4. Do not skip to labels, generation or extraction. GPU-dependent work stays paused.

## Stage 4 closure

Fallback run 20260914T190027Z-stage4fallback-6e78475 is independently verified: 6,474/6,474 units, 20,996,423,633 bytes, zero checksum failures/missing/duplicates/unresolved, source manifest SHA-256 caf3ce503ca0260db47301b6e98bae9fa61831c5de8be0782e970ebc2c9f9127. The released-label audit failed the frozen >=0.95 agreement gate (Spearman 0.6217465713409153), so canonical fallback is importance_primary. Labels: 6,474 rows, 6,434 eligible, train-only q90 1.0685654935597562. Recovery provenance records snapshot_download failure at problem_6596/chunk_103/solutions.json, direct hf_hub_download recovery, unchanged revision/paths/run ID, 2,600 revalidated units, one quarantined/regenerated unit, and resume events. Stage 4 tests: 7 passed. Closure commit changes documentation/provenance only after scientific implementation commit 5e92130.

## Stage 4 support-materialization closure (2026-09-14)

The 6,474 directories under `data/processed/stage4/support_units/` are support-output units (each has `result.json` and an atomic SHA-256 `SUCCESS.json`), not the 6,474 raw fallback source units. They contain MiniLM support counts and predate commit `6a9ae6e`; their raw-source input is the independently verified fallback run. All 6,474 support outputs validated and the committed CUDA batched path completed in 197.039 s (3:17.039), with no failures; the run used the pinned MiniLM revision, float32 normalized embeddings, and `<0.8`.

CPU/ CUDA parity on 100 sentences / 10,000 cosine comparisons: max absolute difference 2.980232238769531e-7, median 1.1920928955078125e-7, p99 2.980232238769531e-7, zero classification, similar-count, dissimilar-count, and NO_ANSWER-count disagreements. The kill/resume and selective-corruption test passes (2 tests). The obsolete predecessor `scripts/archive/stage4_materialize_support_obsolete.py` was unreferenced and retained only for provenance; `scripts/stage4_support_units.py` is the active implementation.

Raw-derived sensitivity inputs were materialized at `data/processed/stage4/raw_sensitivity_inputs.parquet` (6,474 rows) with schema/checksum sidecars and JSON provenance. Train-only thresholds are q80=0.6147450059621669, q90=1.0685654935597562, q95=1.490691200555212. K=4 and K=5 convergence masks are recorded per sentence; eligible counts are 4,489 and 4,627 respectively. NO_ANSWER dissimilar-support count is zero in this corpus. The raw units remain retained pending the user's separate deletion authorization.

The user-authorized deletion gate was subsequently satisfied by verified primary and MacBook secondary backups. The exact raw `stage4_fallback/units/` directory was measured at 20,999,110,344 bytes and deleted; its parent run, manifests, processed outputs, caches, and backups were retained. The deletion, backup hashes, and reacquisition identity are recorded in the persistent `cleanup_record.json`. A lightweight read confirmed both Stage 4 processed Parquet tables remain readable.

The MacBook independently verified the latest incremental backup. A final cleanup verification confirmed `stage4_fallback/units/` remains absent, the source manifest and reacquisition provenance remain present, no Stage 5–11 path directly depends on the deleted directory, and the 6,474 support units plus both processed Parquet artifacts remain readable and checksum-valid.
