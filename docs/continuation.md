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

## Stage 5 stopped-run backup checkpoint (2026-09-15; supersedes earlier next actions)

Run `20260914T234335Z-7e8eef81dc-dbac130d`, executed at clean
`8a466f10a5ac041858fe2eb6ca58cef8d84f8082`, stopped at the authorization guard
at 2026-09-15T01:05:19Z. It is incomplete: 32/256 physical completions,
all fork 6 sample indices 0–31, 188,400 generated tokens, one length truncation.
The observed base projection was 36,685.411 seconds / $20.279, exceeding
10 hours / $20. No adaptive generation or Stage 6–7 execution occurred.

Final read-only validation independently checked raw and processed completion
units, seeds, prompt/output hashes, model identity and 152 existing checksum
sidecars. All passed. The original production and supervisor PIDs are absent,
Docker lists no running containers, and nvidia-smi reports 0 MiB used.
The original failure/status records are preserved, including their FAILED state;
this is an operational budget stop, not a completed Stage 5 milestone.

All run data remain under the persistent NFS run directory. Final evidence is
`stopped_run_backup_validation.json`; `checksums.sha256` covers the final run
payload. W&B offline state is included. Its externally linked core log was also
copied into `logs/preserved_wandb_core_debug.log`; transfer uses rsync `-aL`
to materialize diagnostic symlinks as portable files. No source data were deleted.
The checksum inventory includes those logical alias paths.

This closure changes documentation/provenance only, not execution code or frozen
configuration. The run's execution commit remains unchanged. Push the closure
commit, transfer the entire run to `~/preanchor-backups/` on the MacBook, and
verify every checksum there. That A6000-era run is superseded (hardware switch,
then cross-region reconstruction); no repository evidence records an independent
MacBook verification of it, and none is claimed. The current H100 lineage is
protected by the portable bundle recorded below. Do not create SAFE_TO_TERMINATE
on the basis of this section; resume of any run requires reconciliation of the
original code/config/input identity; never bypass it.

## 2026-09-15 remediation handoff

Stage 4 has been regenerated from scratch with the corrected estimand; see
`docs/decisions.md` (remediation entry) and `docs/stage4_validation.md`. Canonical
fallback remains primary (Spearman 0.858 < 0.95). Every Stage 5 output produced before
the fix is marked non-canonical; production generation must restart from a new
benchmark and a measured projection with `PREANCHOR_HOURLY_RATE` set. PR #1 and PR #2
are not merged; the repaired code lives on local `codex/mvp` (not pushed).

## Hardware switch: A6000 → H100 (2026-09-15 ~12:55 UTC)

The canonical A6000 Stage 5 production run `20260915T124322Z-d6cd3c11d8-5568422c`
was stopped cleanly (SIGINT → `StageInterrupted`) at 4/256 units after the user decided to
move to an H100. Marker: `artifacts/runs/<run_id>/SUPERSEDED_HARDWARE_SWITCH.json`. Its units
are valid and retained as evidence; nothing was deleted. vLLM was stopped and both tmux
sessions closed. All A6000 measurements are in `docs/stage5_measurements.md`.

State at hand-off (repo at the commit introducing this entry; PR #1 merged into `main`
as 6ee01ed):

- Persistent root `/lambda/nfs/preanchor/preanchor` was, at that hand-off, Lambda
  filesystem UUID `63166b57-881d-4c14-8d52-3c71802fd86b` (68 GB used: artifacts 20 GB,
  cache 48 GB, data 2.7 MB). Superseded: the H100 was launched in another region, that
  filesystem was not mountable, and the state was reconstructed on
  `8dd29228-2370-430d-836b-a3e9de7cc148` (see `artifacts/reconstruction/`). A future
  filesystem never has to preserve an old UUID: restore the portable bundle instead
  (`docs/backup_restore.md`).
- Canonical Stage 4: run `20260915T120136Z-b1a2dd9ab0-3b77595c`
  (`labels.parquet` SHA-256 `2049a33f…163d763`), raw run `20260915T113709Z-734a9a5a41-5c37eb5c`.
- `.env` is untracked and holds secrets plus `PREANCHOR_HOURLY_RATE`; set the rate to the
  H100 price on the new instance.

Next actions on the H100 (SPEC §8 admission order, nothing skipped):

```bash
scripts/bootstrap_remote.sh                       # persistent FS, uv, docker, doctor
cp <macbook>/.env .                               # then edit PREANCHOR_HOURLY_RATE
uv run preanchor doctor --config configs/mvp.yaml
scripts/start_vllm.sh configs/lambda/h100_pcie_80.yaml   # capture startup log
python -m preanchor.labels.a6000 benchmark --startup <log>   # per concurrency profile
python -m preanchor.labels.a6000 select --benchmarks <ids> --startup <log>
python -m preanchor.labels.a6000 execute --run-id <prod> --startup <log>
python -m preanchor.labels.a6000 finalize --run-id <prod>
```

The runner is named `a6000` but is hardware-agnostic: it reads the GPU, KV capacity, and
throughput from the live server and the startup log, and the selection rule, truncation gate
(5%), and stop rules (10 h / $20) apply unchanged. Use the H100 profile's concurrency values
as candidate benchmarks; the frozen sampling settings do not change.


## Stage 6 smoke closure (2026-09-15, branch `codex/stage6`)

SPEC Stage 6 is complete and documented in `docs/stage6_validation.md` (plan in
`docs/stage6_plan.md`, amendment `docs/amendments/20260915T170000Z.md`). The extraction
code lives in `src/preanchor/model/` and `src/preanchor/features/`; commands are
`preanchor extract-activations` and `preanchor validate-activations`. Stage 7 (all 40
trajectories under `configs/mvp.yaml`) is **not authorised**: it needs the verified
Stage 5 secondary backup receipt, explicit user approval, and a projection regenerated at
the Stage 7 commit. Launch pattern when authorised:

```bash
uv run preanchor doctor --config configs/mvp.yaml
tmux new-session -d -s stage7 'cd ~/preanchor-stage6 && set -a && . .env && set +a && \
  uv run preanchor extract-activations --config configs/mvp.yaml'
uv run preanchor validate-activations --run-id <run> --config configs/mvp.yaml
```

Resume: `extract-activations --config configs/mvp.yaml --resume --run-id <run>` (same
scope; completed units are validated and never rewritten). Stage 7 observables
(`build-dataset`, `validate-features`) remain unimplemented.
## Portable backup and cross-region restore (2026-09-15)

`preanchor backup`/`preanchor restore` and `scripts/preanchor_portable.sh` replace the
per-stage snapshot scripts for all canonical state (`docs/backup_restore.md`). Canonical
runs on the current filesystem, found from manifests rather than copied IDs: Stage 2
fetch `20260915T133816Z-d7d055d8b7-73b4c983` (metadata; payload reacquired), Stage 4 raw
`20260915T134433Z-d7d055d8b7-05a81618` (provenance; 19.6 GB payload reacquired), Stage 4
build `20260915T141929Z-45b6f6e0cc-07aa7070`, Stage 5 benchmarks
`20260915T143802Z-bb9ba695de-eee71ef4` and `20260915T144551Z-bb9ba695de-4b18d0f3`, and the
finalized Stage 5 production run `20260915T144926Z-bb9ba695de-9fca4c21`. The canonical
bundle is tens of MB (MUST 26.7 MB, SHOULD 8.1 MB); model, dataset and MiniLM caches, the
vLLM image and the raw payloads are pinned and reacquired. Routine protection is one
MacBook command: `./scripts/preanchor_portable.sh backup <ssh-host> --for-stage 5`; a
bundle is a verified secondary copy only once that command prints `BACKUP VERIFIED`.
Continuation on a new filesystem: clone, check out the producing commit, recreate `.env`,
`bootstrap_remote.sh`, then `./scripts/preanchor_portable.sh restore <host> --for-stage 5`
from the MacBook and require `RESTORE VERIFIED` with `Ready to continue Stage 5: YES`.

## Verified MacBook secondary copy of the finalized H100 Stage 5 state (2026-09-15)

The finalized H100 Stage 5 lineage (canonical production run `20260915T144926Z-bb9ba695de-9fca4c21`, Stage 3 complete,
Stage 4 complete, Stage 5 finalized) has an independently verified MacBook secondary copy:

- bundle `20260915T163333Z-d565605b95187a71851608d5dd1f31ad94eaffe3ae29674084da1462e83132a2` (kind `canonical`, highest protected stage 5);
- MacBook destination `~/preanchor-backups/portable/`, verification receipt
  `receipts/20260915T163431Z-pull_verification-d565605b9518-62c783c6.json` (also copied back to Lambda under `artifacts/portable/receipts/`);
- 4,282 unique objects and 33,704,252 bytes rehashed on the MacBook; missing objects 0,
  corrupt objects 0, outcome `secondary_copy_verified`, script output `BACKUP VERIFIED`;
- command: `PREANCHOR_REMOTE_CMD='/home/ubuntu/.local/bin/uv run --frozen preanchor'
  ./scripts/preanchor_portable.sh backup preanchor-h100 --for-stage 5` (the override is
  needed because non-interactive SSH does not put `~/.local/bin` on `PATH`).

The earlier statement that nothing on this filesystem had a secondary copy is resolved
for this lineage. The backup prerequisite for proceeding beyond the protected Stage 5
state is satisfied; Stage 6 remains unauthorized until its own gates pass. Physical
location (`8dd29228-…` virtiofs, host, MacBook path) is provenance only and is not part
of the scientific identity of the run.

## Stage 6 operational closure (2026-09-15)

Stage 6 is finalized and its evidence is in the portable store as bundle
`20260915T180804Z-f4cd4b0b8f03569ba33b87502010f3d015214a3157469b56d947faf1d73e6d7d`
(canonical, protected through Stage 6, restore-verified into a scratch root; see
`docs/stage6_validation.md`). The MacBook secondary copy is verified: receipt
`artifacts/portable/receipts/20260915T181939Z-pull_verification-f4cd4b0b8f03-d185f734.json`
(`secondary_copy_verified`, 4,434 objects, 250,021,342 bytes, 0 missing, 0 corrupt,
`BACKUP VERIFIED`). The Stage 6 backup gate is satisfied; the bundle is the immutable one
from producer `128e1d3`. Stage 7 is not
started and needs explicit authorisation plus a projection regenerated at its launch commit.
Branch `codex/stage6` (contains `codex/mvp`); the MacBook wrapper must point
`PREANCHOR_REMOTE_REPO` at a checkout containing `128e1d3`.

## Stage 7 closure (2026-09-15, branch `codex/stage7`)

SPEC Stage 7 is complete and documented in `docs/stage7_validation.md`. Branch
`codex/stage7` was created from the `codex/stage6` merge commit `afd3714` (Stage 6
`542ee91` plus the portable-backup work); implementation `1cfa152`, `3c2d4ec`; every Stage 7
run executed at clean `3c2d4ec`. Production run
`20260915T182306Z-3ab408d80a-7452f1c2`: 40 units, 6,474 rows, 3.31 GB of float16 shards,
index `1c6c9c2f…6e5a`, feature table `f3227c38…064d` published to
`data/processed/stage7/features.parquet` (plus `feature_audit.json`). Commands:
`extract-activations --config configs/mvp.yaml`, `validate-activations`, `build-dataset
--run-id <run>`, `validate-features --run-id <run>`. The Stage 5 MacBook secondary-copy
receipt was validated with `preanchor backup verify` before launch. The `9bad4b36…`
config hash quoted in the Stage 6 documents belongs to `configs/pilot.yaml`; `configs/mvp.yaml`
resolves to `c8414238…`. Stage 8 has not been started. Next gates before Stage 8: a
verified portable backup covering the Stage 7 run (the sibling session extended
`preanchor backup` to Stage 6 on `codex/stage6`; Stage 7 classification is still to be
added, using `STAGE_DEPENDENCIES` in `src/preanchor/backup/classify.py`), then explicit
user authorisation. Probes read `PRIMARY_FEATURES` only and load one layer at a time from
the shards through the `shard_path`/`row_idx` pointers of the feature table.

## Stage 7 operational closure (2026-09-15, branch `codex/stage7`)

Stage 7 is finalized (`docs/stage7_validation.md`). `codex/stage6` (Stage 6 backup support
and its verified MacBook copy) is merged into `codex/stage7` at `480031e`; Stage 7 backup
support is `56aea1b`. Canonical Stage 7 state is in the portable store as bundle
`20260915T191239Z-161f21fc4351fa8a064ee7a79d0750fa511b5d6afd71543930c5247ee8acc0e5`
(canonical, protected through Stage 7, producer `56aea1b`, 4,791 objects, 3.61 GB logical,
357 objects / 3.12 GB incremental over the verified Stage 6 store; restore-verified into
`/home/ubuntu/preanchor-scratch-restore-stage7-b`). **State: Stage 7 portable bundle prepared
and restore-verified; MacBook secondary verification pending.** The user runs, from the
MacBook:

```bash
PREANCHOR_REMOTE_REPO=/home/ubuntu/preanchor-stage7 PREANCHOR_REMOTE_CMD='/home/ubuntu/.local/bin/uv run --frozen preanchor' ./scripts/preanchor_portable.sh backup preanchor-h100 --for-stage 7 --no-prepare
```

Only `BACKUP VERIFIED` plus a `pull_verification` receipt for that bundle satisfies the
Stage 7 backup gate. Two scratch restore roots (`…-stage7`, `…-stage7-b`, 3.6 GB each on the
local disk) await the user's deletion approval. Stage 8 is not started; it needs the
verified Stage 7 secondary copy and explicit authorisation. Stage 8 consumers read
`data/processed/stage7/features.parquet` (primary features through
`observables.PRIMARY_FEATURES`) and load activations one layer at a time through the
table's `activation_run_id`/`shard_path`/`row_idx` pointers.

## Control plane v0.1 (2026-09-15, branch `agent/ops/control-plane-v1-20260915-0bd79efc`)

Base `f6ce263` (the `codex/stage7` closure). Adds `src/preanchor/control/` and the
`preanchor agent ...` / `preanchor lease ...` commands documented in `docs/concurrency.md`;
enforces `gpu:h100` in every model load, `gpu:h100`+`service:vllm` in
`scripts/start_vllm.sh`, and `portable-store:writer` in `backup prepare` / `import-legacy`
(auto-leased by the CLI, so the MacBook wrapper is unchanged). State lives under
`$PREANCHOR_ROOT/control/` and is outside every scientific identity and every portable
bundle. Validated with the CPU suites and a TEST-namespace drill on the real persistent
root (transcript in `$PREANCHOR_ROOT/control/validation/`). From Stage 8 on: one agent
session = one worker branch (`agent/stage8/<role>-<id>`) = one worktree; `codex/stage8` has
one integrator. The branch is pushed for independent review and is not merged; it does
not start Stage 8 and touches no Stage 3–7 artifact. At the time of writing a separate
session was running a CPU-only Stage 8 `fit` from `/home/ubuntu/preanchor-stage8`
(`codex/stage8`, `0d07354`); that worktree, branch and run were left alone.

