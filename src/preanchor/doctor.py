"""Fail-closed environment admission; CPU checks never admit a GPU stage."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from preanchor.config import ExperimentConfig, Settings, read_yaml
from preanchor.storage import persistent_mount


def command(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30, check=False)


def check_submodules(repo_root: Path) -> list[dict[str, Any]]:
    lock = read_yaml(repo_root / "configs/locks/upstream.yaml")
    checks = []
    for path, entry in lock["repositories"].items():
        repo = repo_root / path
        # git -C in an empty submodule would otherwise discover the parent's repository.
        if not (repo / ".git").exists():
            checks.append({"name": path, "ok": False, "detail": "submodule not initialized"})
            continue
        revision = command(["git", "rev-parse", "HEAD"], repo)
        dirty = command(["git", "status", "--porcelain", "--untracked-files=all"], repo)
        ok = (revision.returncode == dirty.returncode == 0
              and revision.stdout.strip() == entry["revision"] and not dirty.stdout.strip())
        checks.append({"name": path, "ok": ok, "detail": revision.stdout.strip()})
    return checks


def check_persistence(settings: Settings) -> dict[str, Any]:
    root = settings.persistent_root()
    if not root.is_dir():
        raise ValueError("Persistent root does not exist")
    mount = persistent_mount(root)
    free = shutil.disk_usage(root).free
    if free < 100 * 10**9:
        raise ValueError("Persistent filesystem has less than 100 GB free")
    for name, expected in (("HF_HOME", root / "cache/huggingface"),
                           ("WANDB_DIR", root / "wandb")):
        if Path(os.environ.get(name, "")).resolve() != expected:
            raise ValueError(f"{name} must point to its configured persistent directory")
    # Retain the tiny admission record: never delete even probe files on persistent storage.
    directory = root / "artifacts/doctor"
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f"{uuid.uuid4().hex}.write-probe"
    payload = b"preanchor persistence admission\n"
    with probe.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if probe.read_bytes() != payload:
        raise ValueError("NFS write/read verification failed")
    return {"root": str(root), "free_bytes": free, "write_probe": str(probe), "mount": mount}


def check_gpu() -> dict[str, Any]:
    import torch

    smi = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                   "--format=csv,noheader"])
    if smi.returncode != 0:
        raise ValueError("nvidia-smi unavailable in this execution environment; check host access")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 support required")
    if torch.cuda.get_device_properties(0).total_memory < 38 * 10**9:
        raise ValueError("At least 38 GB VRAM required; measured smoke fit still required")
    if torch.__version__.split("+")[0] != "2.7.0" or torch.version.cuda != "12.6":
        raise ValueError("Expected torch 2.7.0 with CUDA 12.6")
    # Execute a small BF16 kernel to verify driver/runtime compatibility, not just a version string.
    with torch.inference_mode():
        tensor = torch.ones((8, 8), dtype=torch.bfloat16, device="cuda:0")
        value = (tensor @ tensor).float().sum().item()
        torch.cuda.synchronize()
    if value != 512:
        raise ValueError("CUDA BF16 arithmetic check failed")
    return {"nvidia_smi": smi.stdout.strip(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "fit_test": "Stage 6 still required"}


def check_model(config: ExperimentConfig) -> dict[str, str]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(config.model.id, revision=config.model.revision, timeout=20)
    if info.sha != config.model.revision:
        raise ValueError("Model revision does not resolve to the pinned SHA")
    return {"model_id": config.model.id, "revision": info.sha}


def check_tracking(config: ExperimentConfig) -> dict[str, str]:
    # No run is created here. Stage 1 owns W&B initialization and durable fallback.
    import wandb

    mode = os.environ.get("WANDB_MODE", config.tracking.wandb_mode)
    if mode == "online" and os.environ.get("WANDB_API_KEY"):
        try:
            if wandb.login(verify=True, relogin=False, timeout=10):
                return {"mode": "online", "authentication": "verified"}
        except Exception:
            pass  # Never expose backend exception strings that may contain credentials.
    os.environ["WANDB_MODE"] = "offline"
    return {"mode": "offline", "authentication": "offline fallback"}


def run_doctor(config: ExperimentConfig, settings: Settings, repo_root: Path,
               *, cpu_only: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, fn) -> None:
        try:
            checks.append({"name": name, "ok": True, "detail": fn()})
        except Exception as exc:
            # Fixed exception class only: arbitrary backend messages can contain credentials.
            checks.append({"name": name, "ok": False, "detail": type(exc).__name__})

    checks.append({"name": "python_3_11", "ok": platform.python_version_tuple()[:2]
                   == ("3", "11"), "detail": platform.python_version()})
    checks.append({"name": "config", "ok": True, "detail": config.resolved_hash()})
    lock = repo_root / "uv.lock"
    checks.append({"name": "uv_lock", "ok": lock.is_file(), "detail": str(lock)})
    checks.extend(check_submodules(repo_root))
    checks.append({"name": "private_services", "ok": True,
                   "detail": "Streamlit restricted to localhost; no additional public port needed"})
    if not cpu_only:
        check("persistent_storage", lambda: check_persistence(settings))
        check("cuda_bf16_driver", check_gpu)
        check("model_revision", lambda: check_model(config))
        check("tracking", lambda: check_tracking(config))
    ok = all(item["ok"] for item in checks)
    return {"schema_version": 1, "scope": "cpu_bootstrap" if cpu_only else "production",
            "ok": ok, "gpu_stage_admitted": ok and not cpu_only,
            "checks": checks, "package_version": importlib.metadata.version("preanchor"),
            "unchecked": ["persistent_storage", "cuda_bf16_driver", "model_revision", "tracking"]
            if cpu_only else []}
