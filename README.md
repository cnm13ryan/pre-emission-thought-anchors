# Preanchor

An offline, passive experiment testing whether model activations immediately before a
reasoning sentence predict that sentence's later counterfactual importance.

Read `AGENTS.md`, `IMPLEMENTATION_SPEC.md`, and `research/project-brief.md` in that order.
The specification owns stage gates and scientific choices. This project uses one model,
one public dataset, and 20 problems. The four held-out problems constitute an
**exploratory held-out engineering test**. No experimental result is available yet.

## Bootstrap

```bash
uv python install 3.11
uv sync --frozen --group dev
uv run --frozen preanchor doctor --config configs/mvp.yaml --cpu-only
CUDA_VISIBLE_DEVICES="" uv run --frozen pytest -q tests/unit tests/integration
uv run --frozen ruff check .
```

The base environment is CPU-only. The pinned CUDA 12.6 PyTorch environment is installed
with `uv sync --frozen --extra gpu --group dev` when the GPU execution gate is resolved.
`doctor --cpu-only` does not inspect CUDA, contact Hugging Face or W&B, write to NFS,
or admit GPU work. Full `doctor` verifies those additional prerequisites without loading
model weights; a measured extraction smoke test is still required for fit.

## Paths and credentials

Code is checked out at `/home/ubuntu/preanchor-repo`. That root-disk checkout is not the
authoritative experiment store. Keep every run, dataset, model cache and log beneath
`PREANCHOR_ROOT=/lambda/nfs/preanchor/preanchor`. See
[environment deviations](docs/environment_deviations.md).

Copy `.env.example` to untracked `.env` and restrict it with `chmod 600 .env`.
Export its values in the launch shell; configuration files never contain credentials.
The settings loader reads the environment. Never expose tokens through commands or logs.

Run `doctor` before GPU stages only after host/sandbox GPU access has been agreed with
the user. The user's host SSH `nvidia-smi` works; Codex sandbox access has not been
admitted. No infrastructure management is implemented.

## Implementation status

Stage 0 supplies the package, lock, config loader, structured logging, doctor and pinned
submodules. Stage 1 implements run provenance and resumable artifact I/O; later stages
remain unimplemented. No dataset or model weights have been downloaded. Scientific
config changes require the amendment and approval process in `AGENTS.md` §4.

Stage 1 commands (with the persistent environment exported):

```bash
uv run --frozen preanchor init-run --config configs/smoke.yaml
uv run --frozen preanchor io-smoke --run-id RUN_ID
uv run --frozen preanchor io-smoke --run-id RUN_ID --resume
uv run --frozen preanchor status --root "$PREANCHOR_ROOT"
```

The smoke command writes synthetic format fixtures only. For the complete interruption
exercise, run `scripts/stage1_smoke.py` detached in tmux. It deliberately interrupts its
own worker, resumes missing work, and checks that completed bytes and mtimes are unchanged.
Run-root success and termination markers are reserved for the complete MVP; Stage 1
completion appears only under `stages/stage1/`.
