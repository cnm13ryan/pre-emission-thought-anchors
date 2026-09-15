"""Content-addressed object store: one immutable payload per SHA-256, never overwritten."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from preanchor.ids import file_hash
from preanchor.io import sync_directory

HEX = frozenset("0123456789abcdef")


def is_digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


class ObjectStore:
    """``objects/sha256/<aa>/<bb>/<sha256>`` beneath a portable root."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.objects = self.root / "objects" / "sha256"

    def path(self, digest: str) -> Path:
        if not is_digest(digest):
            raise ValueError("Object names must be SHA-256 hex")
        return self.objects / digest[:2] / digest[2:4] / digest

    def has(self, digest: str) -> bool:
        path = self.path(digest)
        return path.is_file() and not path.is_symlink()

    def size(self, digest: str) -> int:
        return self.path(digest).stat().st_size

    def verify(self, digest: str) -> int:
        """Independently rehash a stored object; return its byte size."""
        path = self.path(digest)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing object {digest}")
        if file_hash(path) != digest:
            raise ValueError(f"Corrupt object {digest}")
        return path.stat().st_size

    def put_file(self, source: Path, digest: str | None = None) -> tuple[str, bool]:
        """Copy ``source`` into the store. Returns (digest, created).

        An existing object is never replaced; its bytes are re-verified against the
        name instead, so a different payload under the same name is impossible to
        introduce silently.
        """
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Refusing to ingest a symlink or non-file: {source}")
        actual = file_hash(source)
        if digest is not None and digest != actual:
            raise ValueError(f"Source changed while ingesting: {source}")
        target = self.path(actual)
        if target.exists():
            if target.is_symlink() or file_hash(target) != actual:
                raise ValueError(f"Existing object does not match its name: {actual}")
            return actual, False
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if file_hash(temporary) != actual:
            temporary.unlink()
            raise ValueError(f"Copy verification failed: {source}")
        try:
            os.link(temporary, target)   # never clobbers a concurrent identical put
        except FileExistsError:
            temporary.unlink()
            if file_hash(target) != actual:
                raise ValueError(f"Existing object does not match its name: {actual}") from None
            return actual, False
        temporary.unlink()
        sync_directory(target.parent)
        return actual, True

    def install(self, digest: str, destination: Path) -> str:
        """Atomically materialise an object at ``destination``.

        ``restored`` for a new file, ``already_present_identical`` when the destination
        already holds these exact bytes. A different existing file is never overwritten.
        """
        if destination.is_symlink():
            raise ValueError(f"Refusing to write through a symlink: {destination}")
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"Destination is not a regular file: {destination}")
            if file_hash(destination) == digest:
                return "already_present_identical"
            raise FileExistsError(f"Destination holds different bytes: {destination}")
        self.verify(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
        with self.path(digest).open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if file_hash(temporary) != digest:
            temporary.unlink()
            raise ValueError(f"Restore copy verification failed: {destination}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            temporary.unlink()
            if file_hash(destination) == digest:
                return "already_present_identical"
            raise FileExistsError(f"Destination holds different bytes: {destination}") from None
        temporary.unlink()
        sync_directory(destination.parent)
        return "restored"

    def digests(self) -> set[str]:
        if not self.objects.is_dir():
            return set()
        return {p.name for p in self.objects.glob("*/*/*") if p.is_file() and is_digest(p.name)}
