# Concurrency and operational ownership (control plane v0.1)

Several coding agents can work on this repository at once. Git worktrees isolate their
*source* trees, and nothing else: they still share the integration branches, the one
H100, the `preanchor-vllm` container, `$PREANCHOR_ROOT` (`/lambda/nfs/preanchor/preanchor`),
the canonical publication paths under `data/processed/`, and the portable backup store
`artifacts/portable/`. Two agents that each believe they own the GPU, or that both
repoint `artifacts/portable/latest.json`, corrupt operational state without ever touching
the same file in Git. The control plane gives that shared runtime state an explicit,
persistent owner. It is operational metadata only: nothing in it enters a run manifest,
a scientific or resolved config hash, a unit key, a run ID or a portable-bundle identity
(`scientific_identity_hash`), and `artifacts/portable` bundles never include
`control/`.

Implementation: `src/preanchor/control/` (`sessions.py`, `leases.py`, `receipts.py`,
`cli.py`); tests `tests/unit/test_control.py`, `tests/integration/test_control_enforcement.py`.

## Git policy from Stage 8 on

```text
codex/stage8                          integration branch; one owner only (the integrator)

agent/stage8/implementation-<id>      one worker branch per agent session
agent/stage8/tests-<id>
agent/stage8/scientific-review-<id>
agent/stage8/backup-<id>
agent/ops/<topic>-<id>                cross-stage operational work (this control plane)
```

- **One agent session = one worker branch = one worktree.** Register the session
  (`preanchor agent register --worktree <path> ...`) so `preanchor agent status` shows who
  owns which branch and checkout.
- Worker agents never force-reset, rebase or push the integration branch. Only the
  designated integrator (the user, or one explicitly designated session) incorporates
  worker commits, by merge or cherry-pick, after review.
- Competing implementations use separate worker branches *and* separate artifact
  namespaces (distinct run IDs, no shared scratch directories).
- Historical branches `codex/stage5`, `codex/stage6`, `codex/stage7` (and `codex/mvp`,
  `codex/portable-backup`) are never rewritten. This policy is not retrofitted onto
  history; it applies to new work.

## State layout

```text
$PREANCHOR_ROOT/control/
    sessions/<session_id>/session.json      replaceable current view of one session
    sessions/<session_id>/heartbeat.json    replaceable
    leases/<safe-resource>/owner.json       written once per ownership; token hash only
    leases/<safe-resource>/heartbeat.json   replaceable
    leases/.archive/<safe-resource>.<released|reclaimed>.<id>/   ended leases (evidence)
    receipts/<utc>-<kind>-<subject>-<id>.json (+ .sha256)        immutable audit trail
    validation/                              raw validation transcripts (this document)
```

`<safe-resource>` replaces `:` with `__` (`gpu:h100` -> `gpu__h100`). Resource and session
names must match `[A-Za-z0-9][A-Za-z0-9._-]*` segments joined by at most three `:`;
anything else, `..`, `/`, or a symlink anywhere under `control/` fails closed.

## Sessions

```bash
uv run --frozen preanchor agent register --task "<what>" --role <implementation|tests|review|backup|...> --worktree <path>
uv run --frozen preanchor agent heartbeat --session <id> [--note ...]
uv run --frozen preanchor agent status [--stale-after 300] [--include-closed]
uv run --frozen preanchor agent close --session <id> [--outcome ...]
uv run --frozen preanchor agent receipts [--kind ...] [--subject ...]
```

A session record holds `session_id`, `task`, `role`, `hostname`, `pid`, `worktree`,
`branch`, `base_commit`, `current_commit`, `allowed_write_roots`, `forbidden_write_roots`,
`status`, `created_at`, `heartbeat`. Registration and close write immutable receipts;
heartbeat refreshes the current commit. A heartbeat older than `--stale-after`
(default 300 s) shows as `stale`; stale means "look", never "dead", and nothing is ever
closed or reclaimed automatically. Closing a session touches only that session. Sessions
never describe scientific results; run manifests do.

## Leases

```bash
uv run --frozen preanchor lease list
uv run --frozen preanchor lease acquire <resource> --session <id> --purpose "<why>" [--env]
uv run --frozen preanchor lease heartbeat <resource> [--token ...]
uv run --frozen preanchor lease release <resource> [--token ...] [--reason ...]
uv run --frozen preanchor lease reclaim <resource> --session <id> --reason "<evidence>" [--force]
uv run --frozen preanchor lease check <resource>...
```

Resources are free-form names; v0.1 uses:

| Resource | Protects | Enforced by |
| --- | --- | --- |
| `gpu:h100` | the single GPU | `model.load.load_model` (every production model load: extraction, the scientific invariance test) and `features.observables.load_head` on a non-CPU device, checked before `torch` is imported |
| `service:vllm` | the `preanchor-vllm` container | `scripts/start_vllm.sh`, together with `gpu:h100`, before any Docker call; `stop_vllm.sh` needs no lease |
| `portable-store:writer` | `artifacts/portable/{objects,manifests,latest.json}` | `backup.prepare.prepare` and `backup.legacy.import_legacy` before the first store write |
| `canonical-publish:stage<N>` | publishing to `data/processed/stage<N>` | v0.2 (primitive only) |
| `stage-finalize:stage<N>` | closing a stage | v0.2 (primitive only) |

Semantics:

- **Atomic single ownership.** Acquisition is one `mkdir` of the lease directory; on the
  persistent virtiofs root exactly one of N concurrent creators succeeds (verified with
  eight concurrent processes in the tests and six on the real filesystem). The winner then
  writes `owner.json` (resource, session, hostname, pid, worktree, branch, commit, purpose,
  acquired_at, `owner_token_sha256`) and a receipt. An acquisition interrupted between
  `mkdir` and `owner.json` leaves an `incomplete` lease that nobody else can take; it
  needs `reclaim`.
- **Token proof.** `acquire` prints a random 64-hex `owner_token` once. Export it as
  `PREANCHOR_LEASE_TOKENS="gpu:h100=<token>,service:vllm=<token>"` (or `--token`). Only
  the token heartbeats or releases the lease; `owner.json` stores its hash only.
- **Refusal shows the owner.** A contended `acquire` exits 3 and prints the owner session,
  branch, worktree, host/pid, purpose and heartbeat age. A gated command without a valid
  token exits 4 (`LeaseRequired`) before doing anything.
- **Stale is not free.** A stale heartbeat changes nothing about ownership. `nvidia-smi`
  and process tables are telemetry for the operator, never the authority.
- **Reclaim** ends someone else's lease with an immutable `lease_reclaimed` receipt
  carrying the evidence (lease heartbeat age, owner session status and heartbeat age,
  same-host pid liveness) and the mandatory `--reason`. It never transfers ownership; the
  reclaimer acquires afterwards like anyone else. Without `--force` it needs positive
  evidence: the owner session is closed, or the lease is an `incomplete` acquisition
  older than `--stale-after`. A fresh heartbeat, or an owner process that is alive on
  this host, is always refused. A stale lease with no positive evidence is ambiguous and
  needs `--force` plus a reason that records what the operator checked on the owner host.
- Release and reclaim move the lease directory into `leases/.archive/` in one `rename`,
  so the resource becomes free atomically and the old `owner.json` is kept as evidence.

### Working with the GPU

```bash
S=$(uv run --frozen preanchor agent register --task "Stage N extraction" --role implementation --worktree "$PWD" --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["session_id"])')
eval "$(uv run --frozen preanchor lease acquire gpu:h100 --session "$S" --purpose "extract-activations run X" --env)"
uv run --frozen preanchor extract-activations ...          # load_model checks the lease
uv run --frozen preanchor lease release gpu:h100 --reason "run X complete"
```

For vLLM add `service:vllm` (`--env` appends to the existing export) before
`scripts/start_vllm.sh`; release both after `scripts/stop_vllm.sh` confirms the memory is
back. Heartbeat long leases (`preanchor lease heartbeat gpu:h100`) from the driving
session, for example every few minutes from the tmux launcher, so `status` stays honest.

### Portable backups

The MacBook workflow is unchanged:

```bash
./scripts/preanchor_portable.sh backup <ssh-host> --for-stage N
```

Its remote `backup prepare` registers a short-lived `backup-writer` session on Lambda,
takes `portable-store:writer` for the duration of the command, releases it and closes the
session, all with receipts. A concurrent writer fails fast (exit 3, owner shown) before
any object, manifest or `latest.json` write. A caller that already holds the lease exports
its token and no session is created. `BACKUP VERIFIED` is still printed only after the
MacBook rehashed every object. Read-only commands (`backup plan|list|verify|store-path`,
`restore plan`) need no lease. The MacBook-initiated `restore` pushes objects with
`rsync --ignore-existing` (content-addressed, never clobbering) outside the CLI; leasing
that path is deferred.

## Deliberately deferred

**v0.2** — machine-readable stage-transition state; immutable stage-transition receipts;
production launch checkpoints/tags; `preanchor stage close`; evidence-based Stage 7/8
closure; real use of `canonical-publish:stage<N>` (in `build-dataset --publish` and future
publishers) and `stage-finalize:stage<N>`; classification of `control/receipts/` as
operational provenance in portable bundles (today `control/` is simply outside the bundle).

**v0.3** — canonical publication barriers/pointers; independent verifier workflow
integration; one-shot test-evaluation protocol enforcement; richer dashboards only if
shown necessary.

Not planned: daemons, services, databases, schedulers or a general orchestration
framework. The control plane is files, receipts and two CLI groups.

## Validation record (2026-09-15)

Automated: `tests/unit/test_control.py` (21) and
`tests/integration/test_control_enforcement.py` (4, real processes) within the full
CPU suite (`tests/unit tests/integration`: 286 passed, 1 skipped; `tests/scientific`:
5 passed, 1 GPU-only skipped). Real persistent filesystem, TEST namespace only
(`test:control-plane-*` resources, `control-plane-test` sessions): registration, status,
acquire, contention from a second session (exit 3), six simultaneous acquires with one
winner, lease and session heartbeats, wrong-token release refused (exit 4), release,
stale drill (no implicit takeover, reclaim refused without evidence, refused without a
reason, forced reclaim with reason and receipt), close. Transcript and script:
`$PREANCHOR_ROOT/control/validation/20260915T194750Z-control-plane-v0.1-test-namespace.{log,sh}`;
receipts under `$PREANCHOR_ROOT/control/receipts/`. No real resource (`gpu:h100`,
`service:vllm`, `portable-store:writer`, publication, finalization) was leased or
reclaimed. Canonical Stage 3/4/7 tables and `artifacts/portable/latest.json` hashed
identically before and after; the only files that changed under `artifacts/runs` belong
to the other session's live Stage 8 `fit` run, which this work did not touch.
