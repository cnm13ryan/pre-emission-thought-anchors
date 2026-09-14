# Environment deviations

Record verified differences from the planned environment and original/replacement lock hashes if packages change.

## Stage 0 — 2026-09-14 (base commit `85a5604`)

- **Code path (fact):** checkout is `/home/ubuntu/preanchor-repo` on the root disk.
  `/home/ubuntu/preanchor` is a Lambda-created symlink to `/lambda/nfs/preanchor`.
  The user's specified experiment root remains `/lambda/nfs/preanchor/preanchor`.
  Source milestones must also be copied to durable storage; runs and valuable artifacts
  must never default to the checkout.
- **GPU tooling constraint (fact):** `nvidia-smi` from the Codex sandbox returned status 9
  and could not communicate with NVIDIA. The user confirms normal host SSH runs it
  successfully. This is an execution-environment constraint, not evidence of a broken
  host driver. GPU work is paused until the user and agent determine appropriate host
  or sandbox permissions. No driver, CUDA or infrastructure changes are warranted.
- **Sandbox permissions (fact):** Git metadata is read-only in the sandbox. Local
  submodule additions and revision pins require command escalation; nothing is pushed.
  Sandbox DNS blocks dependency downloads, so Python/uv network operations use approved
  escalation. Persistent NFS is readable but outside the sandbox's writable roots.
- **Mount exposure (fact, observed after `2852b60`):** `findmnt` reports the user's
  persistent filesystem at `/lambda/nfs/preanchor` with source
  `b56cf504-e1e5-4726-9de4-1d2e39f756b4` and type `virtiofs`. Its options are
  `ro,nosuid,nodev,relatime` inside the sandbox and `rw,nosuid,nodev,relatime` under
  approved escalation. The first backup attempt rejected the type before creating any
  snapshot. This exposed an overly restrictive implementation check, not a SPEC conflict.
  The guard now accepts native NFS or the Codex shared-filesystem exposure, requires a
  dedicated writable mount beneath `/lambda/nfs`, and verifies copied bytes. It reports
  the observed type without claiming to have inspected the host's backing mount.
  Regression tests reject root-disk, unrelated-mount and read-only cases. No scientific
  choice, resolved config hash or experiment run was changed; no run exists yet.
- **Regenerable tooling (choice):** Python 3.11 is installed in
  `/tmp/preanchor-python`; uv uses `/tmp/preanchor-uv-cache`; `.venv` is ignored in the
  checkout. These are disposable development dependencies, not experiment state.
- **Environment pins:** `uv.lock` resolves the prescribed package versions, including
  torch 2.7.0 from CUDA 12.6. Stage 0 installs only base and dev dependencies. The GPU
  extra has not been installed or tested on the host; no package deviation has been made.

The original full three-document read preceded all repository changes. `AGENTS.md`
remains untouched. The first bootstrap lock hash and validation evidence are recorded
in `docs/stage0_validation.md` after checks complete.

## Stage 1 — permissions update

The tool permission profile now allows filesystem and network access without command
escalation. This does not resolve or waive the user's GPU-access gate: Stage 1 uses only
CPU I/O. No GPU-dependent stage or full GPU doctor has been run. Python and uv continue
using the existing regenerable development locations; `uv.lock` is unchanged.
