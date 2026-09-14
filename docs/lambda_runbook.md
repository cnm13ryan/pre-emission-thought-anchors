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
