# Lambda runbook

Document manual instance setup, persistent storage, private services, recovery and verified backup.

Current checkout: `/home/ubuntu/preanchor-repo`; experiment root:
`/lambda/nfs/preanchor/preanchor`. Never manage Lambda infrastructure or terminate the host.

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
This is a stage-specific backup implementation. Later dataset/model stages have not
been admitted or implemented, and termination markers remain prohibited.

Stage 2 now provides pinned `fetch --discover`, `fetch [--resume --run-id ID]`,
`fetch --smoke`, and `validate-download --run-id ID`. Explicit run IDs prevent ambiguous
resume. Code/config/input identities must match; completed historical runs are not
resumable under a new commit. The stage backup helper is
`scripts/prepare_stage2_backup.py FULL_RUN_ID SMOKE_RUN_ID`; it retains both runs and
minimal inspection evidence, excluding HF caches. See `docs/stage2_validation.md` for
current identifiers. Await the MacBook verification receipt before Stage 3.

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
