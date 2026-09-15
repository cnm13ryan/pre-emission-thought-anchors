# Lambda runbook

Document manual instance setup, persistent storage, private services, recovery and verified backup.

Current checkout: `/home/ubuntu/preanchor-repo`; experiment root:
`/lambda/nfs/preanchor/preanchor`. Never manage Lambda infrastructure or terminate the host.

Fresh instance setup is one command after cloning: `bash scripts/bootstrap_remote.sh`
(idempotent; see the README section "Fresh remote instance"). It installs only missing
tools, syncs the locked GPU environment, creates `.env` and the persistent directories,
and runs the doctor and CPU tests. Rerun it after reboots or pulls; `--quick` skips the
test suite and production doctor. An unmounted persistent filesystem fails the bootstrap.

The user reports working `nvidia-smi` through normal host SSH. Codex's sandbox cannot
communicate with it. Stop before any GPU stage and agree with the user how to execute
the full doctor, `nvidia-smi`, and `torch.utils.collect_env` checks. CPU-only doctor
success does not satisfy GPU admission or the Stage 6 measured fit gate.

Generation and Streamlit must bind to localhost and be accessed through SSH forwarding.
Long jobs run in tmux only after stage, runtime, cost and persistence gates pass.
The user selected the MacBook as the secondary destination: `~/preanchor-backups/`.
The MacBook initiates SSH pulls. Never SSH or push from Lambda to the MacBook.
Prepare an immutable source-only Stage 0 snapshot with
`python scripts/prepare_bootstrap_backup.py` after the bootstrap commit and branch creation.
The snapshot includes `pull_artifacts.sh`; run this script on the MacBook, passing the
Lambda SSH host alias and the exact snapshot path. It prints a verification receipt
only after all copied files match their SHA-256 manifest. Return that receipt to the
agent to record verification. A prepared NFS snapshot alone is not a verified second copy.

The GPU hourly rate/runtime/budget remain unspecified.

Stage 1 provides `preanchor init-run`, `io-smoke --resume`, and `status`. The detached
`scripts/stage1_smoke.py` exercises interruption recovery on tiny synthetic CPU fixtures.
After it passes and the source is committed, prepare its immutable MacBook pull snapshot
with `.venv/bin/python scripts/prepare_stage1_backup.py RUN_ID`. Run the snapshot's
`pull_artifacts.sh` on the MacBook exactly as for Stage 0; return the emitted receipt.
This is a stage-specific backup implementation, retained for Stage 0-2 history; routine
stage-close backups now use the portable workflow in `docs/backup_restore.md`.
Termination markers remain prohibited.

Stage 2 now provides pinned `fetch --discover`, `fetch [--resume --run-id ID]`,
`fetch --smoke`, and `validate-download --run-id ID`. Explicit run IDs prevent ambiguous
resume. Code/config/input identities must match; completed historical runs are not
resumable under a new commit. The stage backup helper is
`scripts/prepare_stage2_backup.py FULL_RUN_ID SMOKE_RUN_ID`; it retains both runs and
minimal inspection evidence, excluding HF caches. See `docs/stage2_validation.md` for
current identifiers (historical; superseded by the portable workflow below).

## A6000 session (2026-09-15)

Host access now works. GPU is RTX A6000 (49,140 MiB), driver 580.105.08, torch
2.7.0+cu126. User planning rate is $1.09/GPU-hour. Prior A100 projection is obsolete.
Use `python -m preanchor.labels.a6000` for this session; old `labels.pilot` retains
historical A100 admission and must not launch this production run.
Server concurrency is `PREANCHOR_REQUEST_CONCURRENCY` (initial 2, eligible 3).
The runner requires matching server/client concurrency and recorded startup log.
Benchmark -> select (immutable A6000 projection) -> execute explicit production
run ID -> stop vLLM -> finalize. All long commands are launched in tmux with NFS
logs. `execute` validates and resumes raw completion units; it never schedules
adaptive indices. No Stage 6 command is authorized.

## Stage 4 commands (2026-09-15)

```bash
uv run --frozen preanchor fetch-raw-resamples --discover        # pins configs/locks/raw_resample_*
uv run --frozen preanchor fetch-raw-resamples --workers 6        # provenance run; --resume --run-id ID
uv run --frozen preanchor validate-raw-resamples --run-id RAW --no-require-identity
HF_HUB_CACHE=$PREANCHOR_ROOT/cache/huggingface/datasets-hub \
  uv run --frozen preanchor audit-upstream --raw-run-id RAW --device cuda
HF_HUB_CACHE=$PREANCHOR_ROOT/cache/huggingface/datasets-hub \
  uv run --frozen preanchor build-labels --run-id AUDIT_RUN --raw-run-id RAW --device cuda
```

Stage 5 production now requires `PREANCHOR_HOURLY_RATE` in `.env`, a completed
benchmark run, and `python -m preanchor.labels.pilot --prepare --benchmark-run-id BENCH`.

## Portable backup (2026-09-15)

Persistent filesystems are region-specific; a new instance may receive an empty
filesystem with a different UUID. Canonical state is protected by the content-addressed
portable store (`docs/backup_restore.md`). Routine, from the MacBook:
`./scripts/preanchor_portable.sh backup <ssh-host> --for-stage <N>` (Stage 6 and later:
from a checkout with Stage 6 backup support, see `docs/backup_restore.md`) after every canonical
stage and before any instance termination; it prints `BACKUP VERIFIED` only after the
MacBook rehashed every object. Restore onto a fresh filesystem with
`./scripts/preanchor_portable.sh restore <new-host> --for-stage <N>` after
`bootstrap_remote.sh`; the remote `restore verify` must print `RESTORE VERIFIED`.
Model, dataset and MiniLM caches, the vLLM image and Stage 2/4 bulk payloads are never
bundled; their pins are in each bundle manifest. The Stage 0-2 snapshot scripts and
`pull_artifacts.sh` remain valid for their historical snapshots and can be ingested with
`preanchor backup import-legacy`.
Non-interactive SSH may not find `uv` in `~/.local/bin`; pass
`PREANCHOR_REMOTE_CMD='/home/ubuntu/.local/bin/uv run --frozen preanchor'`. Future region
migrations restore from the MacBook copy; they never depend on mounting the old
filesystem. Current state: the finalized H100 Stage 5 lineage is verified on the MacBook
(bundle `20260915T163333Z-d565605b95187a71851608d5dd1f31ad94eaffe3ae29674084da1462e83132a2`, see `docs/continuation.md`).
