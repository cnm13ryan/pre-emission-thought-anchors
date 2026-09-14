# Stage 0 validation

Validation date: 2026-09-14. Base commit: `85a5604`. Implementation commit: the Stage 0
bootstrap commit containing this file. This is implementation evidence, not research data.

| Gate | Evidence | Result |
| --- | --- | --- |
| Full instruction read | AGENTS, SPEC lines 1–2228, brief lines 1–314 read before changes | Pass |
| Python and lock | Python 3.11.16; `uv lock`; `uv sync --frozen --group dev` | Pass, no specified package versions changed |
| Unit and integration | `CUDA_VISIBLE_DEVICES='' uv run --frozen pytest -q tests/unit tests/integration` | 18 passed in 0.78s |
| Lint | `uv run --frozen ruff check .` | All checks passed |
| Shell validation | `bash -n scripts/pull_artifacts.sh` | Pass |
| Whitespace validation | `git diff --check` | Pass |
| CPU bootstrap doctor | `uv run --frozen preanchor doctor --config configs/mvp.yaml --cpu-only` | `ok: true`, `scope: cpu_bootstrap`, `gpu_stage_admitted: false` |
| Upstream pins and cleanliness | Doctor checks both submodule HEADs against `configs/locks/upstream.yaml` and checks working trees | Pass |
| Licence inspection | Both pinned `LICENSE.md` files read | MIT notices retained in untouched submodules |
| Mac pull mechanics | Synthetic valid/corrupt copies, repeat destination refusal, remote-shell input rejection | Pass; no real MacBook connection made |
| AGENTS unchanged | SHA-256 below, plus comparison with Git HEAD | Pass |
| Production doctor / model fit | User requires agreement on host/sandbox GPU checks | Not run; GPU admission pending |
| Experiment kill-and-resume | Expensive-unit machinery begins at Stage 1 | Not yet implemented or claimed |
| Second durable backup | User selected MacBook-initiated pulls to `~/preanchor-backups/` | Awaiting actual pull and verification receipt |

Exact hashes:

```text
AGENTS.md SHA-256:
3b93c768240999f64827825af387a3b8b76d77e82eb6dde243c8717067b58fc0
uv.lock SHA-256:
1126f52ae681e23abc08981934af3fc2a43470da54f68ce021a38a044bd55778
Resolved MVP configuration SHA-256:
c841423878148203a22a9492dc8d8c5242ea4d7ec71d7db0538849fcdeb6ef4f
```

Commands in this sandbox additionally set `UV_CACHE_DIR=/tmp/preanchor-uv-cache` and
`UV_PYTHON_INSTALL_DIR=/tmp/preanchor-python`. These locations contain regenerable tools
only. CPU checks do not establish production GPU availability, NFS write admission,
model revision resolution or W&B authentication.

Pytest emits two upstream Typer/Click deprecation warnings about binary/text stream
helpers. The commands work; no dependency changes or warning suppression were made.

No model weights or dataset files were downloaded; no inference, scientific result,
test evaluation, external push or infrastructure change occurred. No `SUCCESS.json`
or `SAFE_TO_TERMINATE` is warranted. Stage 1 must retain the pending backup gate and
the user's explicit stop-before-GPU requirement.
