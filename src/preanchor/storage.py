"""Validate the persistent mount as exposed to the current execution environment."""

import json
import subprocess
from pathlib import Path


def persistent_mount(root: Path) -> dict[str, str]:
    root = root.resolve()
    if not root.is_relative_to("/lambda/nfs") or root == Path("/lambda/nfs"):
        raise ValueError("Persistent root must resolve beneath /lambda/nfs")
    result = subprocess.run(
        ["findmnt", "--json", "--target", str(root), "-o", "SOURCE,FSTYPE,TARGET,OPTIONS"],
        capture_output=True, text=True, timeout=30, check=True)
    mounts = json.loads(result.stdout).get("filesystems", [])
    if len(mounts) != 1:
        raise ValueError("Could not identify the persistent mount")
    mount = mounts[0]
    target = Path(mount["target"])
    if (not target.is_relative_to("/lambda/nfs") or target == Path("/lambda/nfs")
            or not root.is_relative_to(target)):
        raise ValueError("Expected a dedicated filesystem beneath /lambda/nfs")
    # Codex exposes the user-specified Lambda filesystem via virtiofs. See environment deviations.
    if mount["fstype"] not in {"nfs", "nfs4", "virtiofs"}:
        raise ValueError("Unrecognized persistent filesystem exposure")
    if "rw" not in mount["options"].split(","):
        raise ValueError("Persistent mount is read-only in this execution environment")
    return mount
