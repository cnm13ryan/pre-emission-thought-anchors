"""Make an immutable source-only Stage 0 snapshot on NFS; never send it externally."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from preanchor.storage import persistent_mount


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    root = Path(os.environ["PREANCHOR_ROOT"]).resolve()
    if not root.is_relative_to("/lambda/nfs") or not root.is_dir():
        raise SystemExit("An existing persistent PREANCHOR_ROOT under /lambda/nfs is required")
    mount = persistent_mount(root)
    if git(repo, "status", "--porcelain"):
        raise SystemExit("Commit and validate the source before preparing a bootstrap snapshot")
    commit = git(repo, "rev-parse", "HEAD")
    snapshot = root / "artifacts/bootstrap" / f"stage0-{commit[:12]}"
    snapshot.mkdir(parents=True, exist_ok=False)
    bundles = [(repo, "preanchor.bundle", "--all"),
               (repo / "vendor/thought-anchors", "thought-anchors.bundle", "HEAD"),
               (repo / "vendor/thought-anchors.com", "thought-anchors.com.bundle", "HEAD")]
    for source, name, revision in bundles:
        git(source, "bundle", "create", str(snapshot / name), revision)
        git(source, "bundle", "verify", str(snapshot / name))
    # The git bundles include only Git history. Explicitly copy a small reviewable allow-list.
    for relative in ["docs/stage0_validation.md", "docs/environment_deviations.md",
                     "scripts/pull_artifacts.sh"]:
        destination = snapshot / Path(relative).name
        with destination.open("xb") as stream:
            stream.write((repo / relative).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    entries = {}
    for path in sorted(snapshot.iterdir()):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries[path.name] = digest
    manifest = snapshot / "checksums.json"
    with manifest.open("x") as stream:
        json.dump(entries, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Check the NFS copy after writing. A second-copy receipt must come from the MacBook.
    for name, expected in entries.items():
        with (snapshot / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise SystemExit(f"NFS checksum verification failed: {name}")
    print(json.dumps({"snapshot": str(snapshot), "commit": commit,
                      "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "persistent_copy_verified": True, "mount": mount,
                      "secondary_verified": False}, indent=2))


if __name__ == "__main__":
    main()
