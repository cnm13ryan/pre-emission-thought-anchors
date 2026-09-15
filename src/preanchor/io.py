"""Durable atomic artifacts and checksum-validated immutable work units."""

from __future__ import annotations

import base64
import contextlib
import fcntl
import json
import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

from preanchor.ids import file_hash


def json_bytes(value: object) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def lock(path: Path) -> Iterator[None]:
    """Process-scoped advisory exclusion, released by the OS even on SIGKILL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another worker holds {path.name}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_bytes(path: Path, data: bytes, *, replace: bool = False) -> str:
    """Call under the owning unit/run lock. Replacement is for live metadata only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refusing to write through a symlink")
    if path.exists() and not replace:
        if path.read_bytes() == data:
            return file_hash(path)
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)
    return file_hash(path)


def write_artifact(path: Path, data: bytes) -> str:
    digest = atomic_bytes(path, data)
    atomic_bytes(Path(str(path) + ".sha256"), (digest + "\n").encode())
    return digest


def write_json(path: Path, value: object) -> str:
    return write_artifact(path, json_bytes(value))


def write_jsonl(path: Path, rows: list[dict]) -> str:
    return write_artifact(path, b"".join(json_bytes(row) for row in rows))


def read_json(path: Path) -> object:
    validate_artifact(path)
    return json.loads(path.read_bytes())


def validate_artifact(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or unsafe artifact: {path.name}")
    digest = file_hash(path)
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_symlink() or sidecar.read_text().strip() != digest:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return digest


def write_parquet(path: Path, table) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    digest = write_artifact(path, sink.getvalue().to_pybytes())
    write_json(Path(str(path) + ".schema.json"), {
        "schema_version": 1, "format": "arrow_ipc_schema_base64",
        "schema": base64.b64encode(table.schema.serialize().to_pybytes()).decode(),
        "row_count": table.num_rows, "data_sha256": digest})
    validate_parquet(path)
    return digest


def validate_parquet(path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    digest = validate_artifact(path)
    metadata = read_json(Path(str(path) + ".schema.json"))
    expected = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(metadata["schema"])))
    table = pq.read_table(path)
    if (not table.schema.equals(expected, check_metadata=True)
            or table.num_rows != metadata["row_count"] or digest != metadata["data_sha256"]):
        raise ValueError("Parquet schema, grain or checksum mismatch")
    return table


def write_safetensors(path: Path, tensors: dict) -> str:
    from safetensors.numpy import save

    return write_artifact(path, save(tensors))


class UnitStore:
    """Only SUCCESS plus matching content hashes makes a unit reusable."""

    def __init__(self, root: Path):
        self.root = root

    def validate(self, key: str) -> dict:
        directory = self.root / key
        completion = read_json(directory / "SUCCESS.json")
        if completion["key"] != key or completion["schema_version"] != 1:
            raise ValueError("Unit identity mismatch")
        recorded = completion["files"]
        actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
        if actual != set(recorded) | {"SUCCESS.json", "SUCCESS.json.sha256"}:
            raise ValueError("Unit has unrecorded or missing files")
        for name, digest in recorded.items():
            path = directory / name
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Unsafe unit path")
            if file_hash(path) != digest:
                raise ValueError(f"Corrupt unit file: {name}")
            if path.suffix == ".parquet":
                validate_parquet(path)
        return completion

    def produce(self, key: str, writer: Callable[[Path], None]) -> tuple[Path, bool]:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Unit keys must be SHA-256 hex")
        with lock(self.root / ".locks" / key):
            directory = self.root / key
            if directory.exists():
                try:
                    self.validate(key)
                    return directory, False
                except (ValueError, OSError, KeyError, TypeError):
                    quarantine = self.root / "quarantine"
                    quarantine.mkdir(exist_ok=True)
                    os.rename(directory, quarantine / f"{key}-{uuid.uuid4().hex}")
                    sync_directory(quarantine)
                    sync_directory(self.root)
            directory.mkdir(parents=True)
            writer(directory)
            files = {str(p.relative_to(directory)): file_hash(p)
                     for p in sorted(directory.rglob("*")) if p.is_file()}
            if not files or any(".partial" in name for name in files):
                raise ValueError("Cannot complete an empty or partial unit")
            write_json(directory / "SUCCESS.json", {"schema_version": 1, "key": key,
                                                     "files": files})
            self.validate(key)
            return directory, True
