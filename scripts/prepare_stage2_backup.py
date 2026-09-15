"""Prepare a source-and-run snapshot for a MacBook-initiated pull; no external writes."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from preanchor.config import Settings
from preanchor.ids import file_hash
from preanchor.io import lock, read_json, sync_directory
from preanchor.manifest import load_manifest
from preanchor.storage import persistent_mount


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("smoke_run_id")
    args = parser.parse_args()
    if not args.run_id or any(
        c not in "0123456789abcdefghijklmnopqrstuvwxyzTZ-" for c in args.run_id
    ):
        raise SystemExit("Invalid run ID")
    root = Settings().persistent_root()
    persistent_mount(root)
    repo = Path(__file__).resolve().parents[1]
    directory = root / "artifacts/runs" / args.run_id
    if not directory.is_dir():
        raise SystemExit("Run does not exist")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if dirty:
        raise SystemExit("Commit validated source before preparing the backup")
    snapshot = root / "artifacts/backups" / f"stage2-{args.run_id}-{source_commit[:8]}"
    with lock(directory / ".run.lock"):
        manifest = load_manifest(directory)
        if manifest["stages"]["stage2"]["state"] != "complete":
            raise SystemExit("Stage 2 is not complete")
        from preanchor.io import UnitStore

        result = read_json(directory / "stages/stage2/source_manifest.json")
        completion = read_json(directory / "stages/stage2/SUCCESS.json")
        if completion["source_manifest_sha256"] != file_hash(
            directory / "stages/stage2/source_manifest.json"
        ):
            raise ValueError("Source manifest mismatch")
        validated = read_json(directory / "stages/stage2/download_validation.json")
        if (
            validated["state"] != "passed"
            or validated["smoke"]
            or validated["source_manifest_sha256"] != completion["source_manifest_sha256"]
            or validated["files"] != len(result["files"])
        ):
            raise ValueError("Full download validation is missing or inconsistent")
        for row in result["files"]:
            UnitStore(directory / "stages/stage2/units").validate(row["unit_key"])
        smoke = root / "artifacts/runs" / args.smoke_run_id
        if smoke.parent != directory.parent or not smoke.is_dir():
            raise ValueError("Invalid smoke run")
        validation = read_json(smoke / "stages/stage2/resume_validation.json")
        if validation["state"] != "passed":
            raise SystemExit("Persistent interruption/resume validation is missing")
        # Evidence runs retain their original code identity; inspect checksummed stage records.
        smoke_sources = read_json(smoke / "stages/stage2/source_manifest.json")
        for row in smoke_sources["files"]:
            UnitStore(smoke / "stages/stage2/units").validate(row["unit_key"])
        snapshot.mkdir(parents=True, exist_ok=False)
        for origin, label in [
            (directory, "run"),
            (smoke, "smoke"),
            (root / "artifacts/bootstrap/stage2-minimal", "minimal"),
        ]:
            for source in sorted(origin.rglob("*")):
                # Skip W&B convenience symlinks; numbered offline directories are copied.
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(origin)
                if any(
                    part in {"cache", ".env", "credentials", "secrets"} for part in relative.parts
                ):
                    raise ValueError("Forbidden cache or credential path in run backup")
                destination = snapshot / label / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as reader, destination.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                    writer.flush()
                    os.fsync(writer.fileno())
                if file_hash(source) != file_hash(destination):
                    raise ValueError("Persistent copy mismatch")
        for source, name, ref in [
            (repo, "preanchor.bundle", "--all"),
            (repo / "vendor/thought-anchors", "thought-anchors.bundle", "HEAD"),
            (repo / "vendor/thought-anchors.com", "thought-anchors.com.bundle", "HEAD"),
        ]:
            subprocess.run(
                ["git", "-C", str(source), "bundle", "create", str(snapshot / name), ref],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(source), "bundle", "verify", str(snapshot / name)],
                check=True,
                capture_output=True,
            )
        shutil.copyfile(repo / "scripts/pull_artifacts.sh", snapshot / "pull_artifacts.sh")
        entries = {}
        for path in sorted(snapshot.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                    entries[str(path.relative_to(snapshot))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        checksums = snapshot / "checksums.json"
        with checksums.open("x") as stream:
            json.dump(entries, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for path in sorted(snapshot.rglob("*"), reverse=True):
            if path.is_dir():
                sync_directory(path)
        sync_directory(snapshot)
        sync_directory(snapshot.parent)
        for name, expected in entries.items():
            if file_hash(snapshot / name) != expected:
                raise ValueError("Snapshot checksum mismatch")
    print(
        json.dumps(
            {
                "snapshot": str(snapshot),
                "files": len(entries),
                "manifest_sha256": file_hash(checksums),
                "persistent_copy_verified": True,
                "secondary_verified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
