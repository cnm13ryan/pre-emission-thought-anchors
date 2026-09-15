# Stage 2 validation

CPU-only selective acquisition and validation are complete. **The MacBook secondary-copy
receipt remains pending.** No Stage 3 work, GPU admission, model inference, test metrics
or scientific finding is claimed.

Execution commit: `700d34899bb6e195d76702bed7acad93e1ad9f3c` (clean at both runs).
The later commit containing this final record fixes worker-timeout propagation, adds
its regression and corrects the selected-problems search record. The raw completed
runs remain bound to their original code identity; do not resume under changed code.
AGENTS, `uv.lock`, frozen science and both pinned submodules are unchanged.

## Results and gates

| Gate | Evidence | Result |
| --- | --- | --- |
| Exact selection | `configs/locks/dataset_files.txt` and checksummed metadata | 306 exact paths; 339,390,844 bytes |
| Primary coverage | Both correct/incorrect layouts for the frozen twenty IDs | 40 base files, 40 labeled files, 40 chunks files, 40 problem files |
| Audit coverage | Correct Problem 4682 only | All 145 contiguous raw forks, each `solutions.json` |
| Other downloads | README included; singleton pinned allowlists | No other model, variant, combined export or whole-repository content download |
| Pinned source selection | `vendor/thought-anchors/selected_problems.json` parsed | 106 records; first twenty equal frozen IDs |
| Minimal example | `artifacts/bootstrap/stage2-minimal/` under PREANCHOR_ROOT | Five source formats inspected; rejected partials retained |
| Unit/integration plus applicable scientific scope | `CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/unit tests/integration tests/scientific` | 61 passed; two existing Typer warnings |
| Lint/whitespace | `.venv/bin/ruff check .`; `git diff --check` | Pass |
| CPU doctor | `preanchor doctor --config configs/mvp.yaml --cpu-only` | Pass for CPU bootstrap only |
| Detached persistent interruption | `scripts/stage2_smoke.py` in tmux | SIGTERM exit 1, SIGKILL exit -9; two partial units quarantined |
| Smoke resume | Five durable units | Completed six-file unit preserves all SHA-256 and ns mtimes; resume creates 4, then 0 |
| Full acquisition | Detached `preanchor fetch --run-id ...` | 306 created, 177.497 seconds, 169,984,000-byte peak recorded RAM |
| Full validation/no-op resume | `download_validation.json` | All upstream/local hashes valid; created 0, skipped 306; all 1,836 completed-unit files unchanged |
| Tracking | Four smoke and two full-run offline sessions | Retained for backup; GPU telemetry disabled by existing gate |
| Secondary copy | Prepared source/run snapshot, MacBook initiates pull | Pending receipt; Stage 3 paused at backup gate |

Only the Stage 2 source-scope scientific tests currently exist; later label, pre-emission,
fit-scope and test-lock tests remain unimplemented and unclaimed.

## Durable evidence

All relative paths below are under `/lambda/nfs/preanchor/preanchor`.

Smoke run: `artifacts/runs/20260914T182859Z-1c085746e3-6c451e8c`.
Full run: `artifacts/runs/20260914T183017Z-1c085746e3-08bc1e7c`.

| Artifact | SHA-256 |
| --- | --- |
| Smoke `stages/stage2/resume_validation.json` | `1368e6de0422b77087905a672a0e0e967d12c367d4063839529bbe00bec86d67` |
| Full `stages/stage2/source_manifest.json` | `ff5c1d6d043db2d1f8a01b2c5152bb8139e17b8f1bd5f11cb360821baee65663` |
| Full `stages/stage2/download_validation.json` | `c867756954d219606418e0d65f0042ecbfa7ec97077fc26024729b0888de809c` |
| Full `stages/stage2/resource_projection.json` | `bdd01515112771edd23cb3e9c9461edf39dc3c5f6731c0e30916d315e57a1316` |

The current-commit uncached 3,235,948-byte fork benchmark took 0.715 seconds. A conservative
fourfold transfer-plus-request projection was 1,174.95 seconds (under one hour), recorded
before the full fetch. Disk estimate includes both source and HF cache copies. No GPU
was used; VRAM is unmeasured. Hourly rate and generation budget remain unspecified;
no monetary cost is invented for the CPU acquisition.

The source manifest contains each exact upstream path, revision context, byte size,
Git/LFS identity, local SHA-256 and durable unit path. Raw files are unchanged. This stage
does not certify sentence alignment, label validity or resampling scientific equivalence.

## Corrections and recovery

Minimal inspection caught two draft API/schema assumptions before scale: Hub types live
in `huggingface_hub.hf_api`; `chunks.json` is an object containing string chunks while
`chunks_labeled.json` is a list of objects. The initial continuation's failed search for
`selected_problems.json` was corrected by reading the pinned source file. No scientific
selection or dependency changed.

Final review distinguished a worker's actual TimeoutError from the polling timeout.
Its regression proves failure recording, rejection of partial completion and successful
resume. No timeout occurred during the completed full acquisition. Historical failures
and quarantine remain intact; current manifest/heartbeat state governs completion.

Prepare the MacBook-pull snapshot with:

```bash
PREANCHOR_ROOT=/lambda/nfs/preanchor/preanchor .venv/bin/python scripts/prepare_stage2_backup.py \
  20260914T183017Z-1c085746e3-08bc1e7c 20260914T182859Z-1c085746e3-6c451e8c
```

It includes source bundles, full and smoke run records, minimal inspection evidence and
W&B offline directories. It excludes HF caches and secrets. The MacBook runs the existing
`pull_artifacts.sh` with its existing Lambda SSH alias, emitted snapshot path and
`~/preanchor-backups/`. No host-side push, external Git action, infrastructure change,
run-root SUCCESS or SAFE_TO_TERMINATE is authorized or performed.
