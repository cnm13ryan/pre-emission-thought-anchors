# Stage 1 validation

Stage 1 implementation is validated on synthetic CPU data. No scientific finding or
GPU admission is claimed. Its secondary backup remains pending the MacBook pull receipt.

Implementation commit: `b002449e87210655813cfd8d78c953f95a803b1b` (clean at execution).
The subsequent status-only correction and regression tests are in the commit containing
this document. `AGENTS.md`, frozen scientific settings and `uv.lock` are unchanged.

## Evidence

Persistent run:
`/lambda/nfs/preanchor/preanchor/artifacts/runs/20260914T175938Z-fab7e8ae04-f1b5dc10`

| Gate | Evidence | Result |
| --- | --- | --- |
| Unit/integration suite | `CUDA_VISIBLE_DEVICES='' uv run --frozen pytest -q tests/unit tests/integration` | 41 passed in 3.11s; same two upstream Typer/Click warnings |
| Lint | `uv run --frozen ruff check .` | Pass |
| CPU doctor | `preanchor doctor --config configs/mvp.yaml --cpu-only` | Pass; GPU admission false |
| Stable IDs | `tests/unit/test_ids.py` | Exact sample seed formula; each cache input changes identity |
| Atomic formats | `tests/integration/test_atomic_cache.py` | JSON, JSONL, Parquet/schema and float16 safetensors round trips |
| Corruption recovery | Atomic-cache tests | Bad bytes retained in quarantine; only corrupt unit rebuilt |
| Resume isolation | `tests/integration/test_manifest.py` | Changed config, code/diff or dependency lock refused |
| Manifest publication | Manifest tests | Current view validated against immutable revision even after interrupted sidecar publication |
| Main-loop heartbeat | Manifest tests | Updates at 30 seconds; completed stale timestamps do not signal a stalled job |
| Real persistent interruption | `stages/stage1/resume_validation.json` | SIGTERM exit 1 and SIGKILL exit -9; two partial units quarantined |
| Immutable completed data | Same validation record | All 12 files of the completed unit retain exact SHA-256 and nanosecond mtime |
| Targeted resume | `stages/stage1/attempts/` and manifest timings | First final resume created 2 units; subsequent resume created 0, skipped 3 |
| Failure evidence | `stages/stage1/failures/` | Graceful failure traceback and recovered unfinished-attempt record retained |
| W&B fallback | Tracking test and four `logs/wandb/offline-run-*` directories | Online failure simulated; real smoke recorded offline |
| Persistent storage | Actual smoke writes and validates on the designated writable mount | Pass, CPU only |

The smoke ran detached in tmux session `preanchor-stage1-b002449`. Its outer log is
also retained as `logs/orchestrator.txt` inside the run. Peak recorded process RAM was
177,500,160 bytes. No GPU was used; peak GPU memory is explicitly unmeasured, not zero.
The SIGKILL attempt cannot provide a final process timing; recovery records that it
ended without a terminal record and does not infer a signal from that fact alone.

Hashes at validation:

```text
stages/stage1/resume_validation.json
89e5701ecf2321687127970f63a21b797057b44de9d927d00a70341d93984c8d

manifest.json (after linking validation evidence)
cb706e599f820ca7856ea02d16b574dff84981eb65adf5da6eaa92b5e34a2ea4

uv.lock
1126f52ae681e23abc08981934af3fc2a43470da54f68ce021a38a044bd55778
```

## Storage and continuation

Each immutable artifact has a checksum sidecar. Parquet also has a serialized Arrow
schema, row count and data checksum in its checksummed schema sidecar. A unit is accepted
only when `SUCCESS.json`, its sidecar and every recorded file validate. Failed or partial
units are renamed into quarantine. No persistent file is deleted by resume.

Run manifests have checksummed immutable revisions; `manifest.json` is the current view.
Heartbeat and the latest-failure view are live metadata; failure history stays immutable.
`failure.json` may describe a historical failed attempt even after successful resume;
the manifest and heartbeat report current stage state.

The completed run remains bound to its original implementation commit. Future code
changes require new run identities rather than resuming it under a different code hash.
The backup includes source Git bundles, all run artifacts, partial-unit evidence and
offline W&B records. It excludes model caches and secrets. The MacBook initiates the pull
into `~/preanchor-backups/`; no host-side push or SSH connection to the MacBook is used.

No Stage 2 import, model download, frozen scientific amendment, test-set evaluation,
external push or infrastructure change occurred. GPU-dependent stages remain paused.
No run-root `SUCCESS.json` or `SAFE_TO_TERMINATE` has been created.
