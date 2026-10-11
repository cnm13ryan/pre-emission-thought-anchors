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

## Stage 4 — GPU access resolution

The current Codex context passed `nvidia-smi` and the locked BF16 CUDA check on the
A100-SXM4-40GB (driver 580.105.08). The earlier failure was an execution-context
restriction, not a host driver failure. No driver, CUDA toolkit or GPU class changed.
Production `preanchor doctor` passed with torch `2.7.0+cu126`; the Stage 6 measured model
fit gate remains unrun and GPU workloads remain paused by user instruction.

## 2026-09-15 A6000 environment collector

Pinned torch 2.7.0+cu126 passes actual BF16 kernel admission. Native
`python -m torch.utils.collect_env` crashes in `get_pip_packages`: uv's locked
virtual environment has no pip, so torch calls `.splitlines()` on None. Preserve
the native command's stdout/stderr, then run the same collector with only package
enumeration supplied by `importlib.metadata`. All GPU/driver/CUDA probes remain
native torch code; no installed library, driver, CUDA or uv.lock is changed.
Both lock hashes are therefore identical. Package provenance is independently
captured in every manifest and environment/packages.json. Alternative rejected:
installing an unneeded, unpinned pip or hiding the collector failure.

## 2026-09-15 remediation session

- **Mount (fact):** the persistent filesystem now mounts with source
  `63166b57-881d-4c14-8d52-3c71802fd86b` (virtiofs, rw), not the
  `b56cf504-...` recorded for Stage 0. Stage 2 (full run) and Stage 3 tables were
  restored; no pre-fix Stage 4 processed artifacts exist here.
- **Root-owned hub cache (fact):** `cache/huggingface/hub` is owned by root because the
  vLLM container mounts it. Dataset transit downloads and the MiniLM model use the
  user-owned `cache/huggingface/datasets-hub` (CLI fallback; `HF_HUB_CACHE` for
  audit/build). No ownership on persistent storage was changed.
- **Rate limiting (fact):** the Hub returned HTTP 429 during the parallel raw fetch;
  the client's retries succeeded and every unit validated. Lock hashes unchanged.
- **Hourly rate:** `PREANCHOR_HOURLY_RATE` is required by Stage 5 projections and is
  currently unset in `.env`.

## Redesign Phase 1 — 2026-10-09 (parent commit `f11823f`, MacBook, CPU/MPS only)

- **Separate analysis environment (fact):** the locked CPU `.venv` has no torch,
  transformers or sentence-transformers, and the GPU extra pins Linux CUDA wheels. The
  exploratory Phase 1 scripts in `research/redesign/` therefore run in an untracked venv at
  `~/preanchor-root/cache/redesign-venv`, with the same package versions as `uv.lock`
  where they exist (torch 2.7.0 macOS wheel, transformers 4.53.2,
  sentence-transformers 5.0.0, pandas 2.3.1, pyarrow 20.0.0, scikit-learn 1.7.1,
  scipy 1.16.0, numpy 2.2.6, huggingface_hub 0.33.4). The freeze is
  `~/preanchor-root/cache/redesign-venv-freeze.txt`. `uv.lock` and `.venv` are unchanged.
  No frozen stage was rerun in this environment.
- **Device differences (fact):** MiniLM similarity runs on MPS rather than the Stage 4
  CUDA device, and layer-48 next-token distributions are recomputed in bf16 on MPS. Both
  are checked against the stored values, and the agreement is reported in
  `research/redesign/phase1_report.md`.
