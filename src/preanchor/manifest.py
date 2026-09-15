"""Immutable manifest revisions with a recoverable current view."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from preanchor.config import ExperimentConfig, config_hash, read_yaml
from preanchor.ids import file_hash
from preanchor.io import atomic_bytes, json_bytes, read_json, write_artifact, write_json
from preanchor.logging import redact


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def provenance(repo: Path) -> tuple[dict, bytes]:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    patch = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--binary"], text=True)
    if redact(patch) != patch:
        raise ValueError("Refusing to persist a diff containing an environment credential")
    untracked = {}
    for name in git("ls-files", "--others", "--exclude-standard").splitlines():
        path = repo / name
        if path.is_file():
            untracked[name] = file_hash(path)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in
                      importlib.metadata.distributions())
    upstream = read_yaml(repo / "configs/locks/upstream.yaml")
    for name, entry in upstream["repositories"].items():
        if (git("-C", name, "rev-parse", "HEAD") != entry["revision"]
                or git("-C", name, "status", "--porcelain")):
            raise ValueError("Upstream revision or working tree does not match the lock")
    diff_hash = config_hash({"patch": patch, "untracked": untracked})
    return {"git_sha": git("rev-parse", "HEAD"), "dirty_diff_hash": diff_hash,
            "dirty": bool(patch or untracked), "untracked_hashes": untracked,
            "lock_hash": file_hash(repo / "uv.lock"), "package_freeze": packages,
            "upstream": upstream, "os": platform.platform(), "python": platform.python_version(),
            "architecture": platform.machine(),
            "hardware": {"gpu": None, "driver": None, "cuda": None, "torch": None,
                         "status": "not_checked_gpu_access_gate_pending"}}, patch.encode()


def identity(config: ExperimentConfig, source: dict, inputs: dict) -> dict:
    return {"resolved_config_hash": config.resolved_hash(),
            "scientific_config_hash": config.scientific_hash(),
            "git_sha": source["git_sha"], "dirty_diff_hash": source["dirty_diff_hash"],
            "lock_hash": source["lock_hash"], "input_hashes": inputs}


def save_manifest(directory: Path, manifest: dict) -> None:
    payload = json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    write_artifact(directory / "manifests" / f"{digest}.json", payload)
    # The immutable revision is durable before the replaceable current view.
    atomic_bytes(directory / "manifest.json", payload, replace=True)
    atomic_bytes(directory / "manifest.json.sha256", (digest + "\n").encode(), replace=True)


def load_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    digest = file_hash(path)
    # A killed publication can leave an old sidecar. Validate against its immutable revision.
    revision = directory / "manifests" / f"{digest}.json"
    manifest = read_json(revision)
    if revision.read_bytes() != path.read_bytes():
        raise ValueError("Manifest current view differs from immutable revision")
    return manifest


def create_run(root: Path, repo: Path, config: ExperimentConfig, commands: list[str]) -> Path:
    source, patch = provenance(repo)
    run_identity = identity(config, source, {})
    run_id = (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-"
              + config_hash(run_identity)[:10] + "-" + uuid.uuid4().hex[:8])
    directory = root / "artifacts/runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    for name in ["environment", "stages", "data", "activations", "probe_models", "predictions",
                 "metrics", "figures", "trajectory_exports", "logs"]:
        (directory / name).mkdir()
    resolved = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True).encode()
    write_artifact(directory / "resolved_config.yaml", resolved)
    write_artifact(directory / "git.patch", patch)
    write_json(directory / "environment/packages.json", source["package_freeze"])
    manifest = {"schema_version": 1, "run_id": run_id, "started_at": utc_now(),
                "identity": run_identity, "source": source,
                "revisions": {"model": config.model.revision,
                              "tokenizer": config.model.tokenizer_revision,
                              "dataset": config.dataset.revision,
                              "embedding": config.labels["similarity_revision"]},
                "vllm": {"image": config.generation["vllm_image"], "digest": None,
                         "status": "not_started"},
                "seed_root": config.seed_root, "sample_seeds": [], "fold_seeds": [],
                "commands": redact(commands), "stages": {}, "stage_timings": {},
                "input_hashes": {}, "output_hashes": {}, "wandb": [],
                "backup": {"destination": "MacBook:~/preanchor-backups/",
                           "verified": False},
                "gpu_gate": "paused_by_user_pending_host_access_resolution"}
    save_manifest(directory, manifest)
    return directory


def check_resume(directory: Path, repo: Path, config: ExperimentConfig) -> dict:
    manifest = load_manifest(directory)
    source, _ = provenance(repo)
    current = identity(config, source, manifest["input_hashes"])
    if manifest["identity"] != current:
        raise ValueError("Resume refused: config, code, dependency lock or input hashes differ")
    from preanchor.io import validate_artifact

    validate_artifact(directory / "resolved_config.yaml")
    if read_yaml(directory / "resolved_config.yaml") != config.model_dump(mode="json"):
        raise ValueError("Resolved configuration differs from the requested configuration")
    return manifest
