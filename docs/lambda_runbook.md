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
