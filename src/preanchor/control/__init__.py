"""Operational control plane v0.1: sessions, persistent leases and audit receipts.

Everything here is operational metadata under ``$PREANCHOR_ROOT/control/``. It never
enters a run manifest, a scientific hash, a unit key or a portable-bundle identity.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

from preanchor.config import Settings

CONTROL_DIRNAME = "control"
STALE_AFTER_SECONDS = 300.0
TOKENS_ENV = "PREANCHOR_LEASE_TOKENS"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}(:[A-Za-z0-9][A-Za-z0-9._-]{0,63}){0,3}$")


class ControlError(ValueError):
    """Any control-plane refusal; the message is safe to show to an operator."""


def control_root(root: Path | None = None) -> Path:
    """``<root>/control``. ``root`` defaults to ``PREANCHOR_ROOT`` (no mount check: the
    control plane must also work while diagnosing a broken mount)."""
    if root is None:
        settings = Settings()
        if settings.root is None:
            raise ControlError("PREANCHOR_ROOT must be set (or pass --root)")
        root = settings.root
    return Path(root).resolve() / CONTROL_DIRNAME


def safe_name(resource: str) -> str:
    """Directory-safe form of a resource or session name; anything else fails closed."""
    if not isinstance(resource, str) or not _NAME.match(resource) or ".." in resource:
        raise ControlError(f"Malformed resource name: {resource!r}")
    return resource.replace(":", "__")


def refuse_symlinks(path: Path, stop: Path) -> None:
    """Refuse a control path if it, or any component beneath ``stop``, is a symlink."""
    for candidate in [path, *path.parents]:
        if candidate == stop.parent:
            break
        if candidate.is_symlink():
            raise ControlError(f"Unsafe symlinked control path: {candidate}")


def hostname() -> str:
    return platform.node()


def git_facts(worktree: Path | None) -> dict:
    """Branch and commit of a checkout (operational provenance; failures are recorded,
    never fatal, because the control plane must work in a detached or broken checkout)."""
    facts = {"worktree": str(worktree.resolve()) if worktree else None,
             "branch": None, "commit": None}
    if worktree is None:
        return facts

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(worktree), *args], capture_output=True,
                                  text=True, timeout=30, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    facts["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
    facts["commit"] = git("rev-parse", "HEAD")
    return facts


def presented_tokens(environ: dict | None = None) -> dict[str, str]:
    """``PREANCHOR_LEASE_TOKENS="gpu:h100=<token>,service:vllm=<token>"``."""
    raw = (environ if environ is not None else os.environ).get(TOKENS_ENV, "")
    tokens = {}
    for item in filter(None, (part.strip() for part in raw.split(","))):
        resource, _, token = item.partition("=")
        if not token:
            raise ControlError(f"Malformed {TOKENS_ENV} entry: {item!r}")
        tokens[resource] = token
    return tokens


def process_alive(pid: int) -> bool | None:
    """True/False for this host; None when it cannot be determined."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True
