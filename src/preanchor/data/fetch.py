"""Exact upstream discovery and immutable, independently resumable source files."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile, RepoFolder

from preanchor.config import config_hash
from preanchor.io import read_json, write_artifact, write_json

PREFIX = "deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95"
AUXILIARY = ["base_solution.json", "chunks_labeled.json", "chunks.json", "problem.json"]


def metadata(entry, role, problem_id=None, trace_kind=None):
    return {
        "path": entry.path,
        "size": entry.size,
        "git_blob_sha1": entry.blob_id,
        "lfs_sha256": entry.lfs.sha256 if entry.lfs else None,
        "role": role,
        "problem_id": problem_id,
        "trace_kind": trace_kind,
    }


def discover(config, *, api=None, progress=lambda: None):
    api = api or HfApi()
    kwargs = dict(repo_id=config.dataset.id, repo_type="dataset", revision=config.dataset.revision)
    records = []
    root = list(api.list_repo_tree(**kwargs, recursive=False))
    for entry in root:
        if isinstance(entry, RepoFile) and entry.path == "README.md":
            records.append(metadata(entry, "dataset_card"))
    audit_forks = []
    for kind in ["correct", "incorrect"]:
        parent = f"{PREFIX}/{kind}_base_solution"
        children = list(api.list_repo_tree(**kwargs, path_in_repo=parent, recursive=False))
        actual = {e.path.rsplit("/", 1)[-1] for e in children if isinstance(e, RepoFolder)}
        expected = {f"problem_{i}" for i in config.dataset.problem_ids}
        if actual != expected:
            raise ValueError(f"Unexpected problem directories in {parent}: {actual ^ expected}")
        for problem in sorted(config.dataset.problem_ids):
            progress()
            directory = f"{parent}/problem_{problem}"
            entries = list(api.list_repo_tree(**kwargs, path_in_repo=directory, recursive=False))
            files = {PurePosixPath(e.path).name: e for e in entries if isinstance(e, RepoFile)}
            for name in AUXILIARY:
                if name not in files:
                    raise ValueError(f"Missing pinned source: {directory}/{name}")
                records.append(metadata(files[name], name.removesuffix(".json"), problem, kind))
            if kind == "correct" and problem == 4682:
                audit_forks = sorted(
                    int(PurePosixPath(e.path).name.removeprefix("chunk_"))
                    for e in entries
                    if isinstance(e, RepoFolder) and PurePosixPath(e.path).name.startswith("chunk_")
                )
                raw = list(api.list_repo_tree(**kwargs, path_in_repo=directory, recursive=True))
                raw_files = [e for e in raw if isinstance(e, RepoFile) and "/chunk_" in e.path]
                if not audit_forks or audit_forks != list(range(len(audit_forks))):
                    raise ValueError("Audit fork directories must be contiguous from zero")
                expected_raw = {f"{directory}/chunk_{i}/solutions.json" for i in audit_forks}
                if {e.path for e in raw_files} != expected_raw:
                    raise ValueError("Unknown audit resample layout; inspect before downloading")
                records.extend(metadata(e, "audit_resamples", problem, kind) for e in raw_files)
    result = {
        "schema_version": 1,
        "dataset_id": config.dataset.id,
        "revision": config.dataset.revision,
        "files": sorted(records, key=lambda r: r["path"]),
        "audit_forks": audit_forks,
        "selected_problems_json": "absent in inspected root and selected trajectory directories",
    }
    validate_discovery(result, config)
    return result


def validate_discovery(discovery, config):
    if (
        discovery["schema_version"] != 1
        or discovery["dataset_id"] != config.dataset.id
        or discovery["revision"] != config.dataset.revision
    ):
        raise ValueError("Dataset identity mismatch")
    forks = discovery["audit_forks"]
    if not forks or forks != list(range(len(forks))):
        raise ValueError("Audit forks must be contiguous from zero")
    if sum(r["role"] == "dataset_card" for r in discovery["files"]) != 1:
        raise ValueError("Exactly one dataset card required")
    paths = [r["path"] for r in discovery["files"]]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("Empty or duplicate allowlist")
    pairs = {(p, k) for p in config.dataset.problem_ids for k in ["correct", "incorrect"]}
    for role in [name.removesuffix(".json") for name in AUXILIARY]:
        rows = [r for r in discovery["files"] if r["role"] == role]
        if len(rows) != 40 or {(r["problem_id"], r["trace_kind"]) for r in rows} != pairs:
            raise ValueError("Primary trajectory coverage must be exactly 20 × 2")
    for row in discovery["files"]:
        path = PurePosixPath(row["path"])
        if path.is_absolute() or ".." in path.parts or any(c in row["path"] for c in "*?[]"):
            raise ValueError("Allowlist entries must be exact safe paths")
        for field, length in [("git_blob_sha1", 40), ("lfs_sha256", 64)]:
            value = row[field]
            if field == "lfs_sha256" and value is None:
                continue
            if (
                not isinstance(value, str)
                or len(value) != length
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Invalid upstream hash")
        if row["size"] < 0:
            raise ValueError("Invalid source size")
        if row["role"] == "dataset_card":
            expected = "README.md"
        else:
            base = f"{PREFIX}/{row['trace_kind']}_base_solution/problem_{row['problem_id']}"
            if row["role"] == "audit_resamples":
                if (row["problem_id"], row["trace_kind"]) != (4682, "correct"):
                    raise ValueError("Only the approved audit trajectory may supply resamples")
                fork = path.parent.name
                if fork not in {f"chunk_{i}" for i in discovery["audit_forks"]}:
                    raise ValueError("Unregistered audit fork")
                expected = f"{base}/{fork}/solutions.json"
            elif row["role"] in {n.removesuffix(".json") for n in AUXILIARY}:
                expected = f"{base}/{row['role']}.json"
            else:
                raise ValueError("Unknown download role")
        if row["path"] != expected:
            raise ValueError("Path is outside the exact selected layout")
    audit = [r for r in discovery["files"] if r["role"] == "audit_resamples"]
    if len(audit) != len(discovery["audit_forks"]):
        raise ValueError("Incomplete audit coverage")


def validate_source(path, record):
    if path.stat().st_size != record["size"]:
        raise ValueError("Upstream file size mismatch")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if record["lfs_sha256"]:
        if digest != record["lfs_sha256"]:
            raise ValueError("Upstream LFS SHA-256 mismatch")
    elif (
        hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        != record["git_blob_sha1"]
    ):
        raise ValueError("Upstream Git blob hash mismatch")
    if path.suffix == ".json":
        value = json.loads(payload)
        role = record["role"]
        if role in {"base_solution", "problem"} and not isinstance(value, dict):
            raise ValueError("Expected source object")
        if role == "chunks_labeled" and not isinstance(value, list):
            raise ValueError("Expected labeled chunk list")
        if role == "chunks" and not (
            isinstance(value, dict) and isinstance(value.get("chunks"), list)
        ):
            raise ValueError("Expected source chunk list")
        if role == "base_solution" and not isinstance(value.get("full_cot"), str):
            raise ValueError("Base solution lacks exact full_cot")
        if role == "audit_resamples" and not isinstance(value, (dict, list)):
            raise ValueError("Expected raw resample collection")
    return digest


def source_key(record, discovery, identity):
    return config_hash(
        {
            "stage": "stage2",
            "dataset_revision": discovery["revision"],
            "record": record,
            "identity": identity,
        }
    )


def fetch_file(
    store,
    record,
    discovery,
    identity,
    cache_dir,
    *,
    download=snapshot_download,
    delay_seconds=0,
    stop_event=None,
):
    key = source_key(record, discovery, identity)
    started = time.monotonic()

    def writer(directory):
        write_json(directory / "request.json", record)
        if stop_event is not None and stop_event.wait(delay_seconds):
            raise RuntimeError("Download interrupted before request")
        snapshot = download(
            repo_id=discovery["dataset_id"],
            repo_type="dataset",
            revision=discovery["revision"],
            allow_patterns=[record["path"]],
            cache_dir=str(cache_dir),
            max_workers=1,
        )
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Download interrupted before publication")
        source = Path(snapshot) / record["path"]
        validate_source(source, record)
        write_artifact(directory / PurePosixPath(record["path"]).name, source.read_bytes())

    directory, created = store.produce(key, writer)
    path = directory / PurePosixPath(record["path"]).name
    digest = validate_source(path, record)
    return {
        **record,
        "unit_key": key,
        "local_path": str(path),
        "sha256": digest,
        "created": created,
        "elapsed_seconds": time.monotonic() - started,
    }


def save_discovery(discovery, repo):
    payload = "".join(r["path"] + "\n" for r in discovery["files"]).encode()
    write_artifact(repo / "configs/locks/dataset_files.txt", payload)
    write_json(repo / "configs/locks/dataset_metadata.json", discovery)


def load_discovery(repo, config):
    value = read_json(repo / "configs/locks/dataset_metadata.json")
    validate_discovery(value, config)
    expected = "".join(r["path"] + "\n" for r in value["files"])
    from preanchor.io import validate_artifact

    path = repo / "configs/locks/dataset_files.txt"
    validate_artifact(path)
    if path.read_text() != expected:
        raise ValueError("Allowlist differs from pinned metadata")
    return value
