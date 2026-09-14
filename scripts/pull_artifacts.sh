#!/usr/bin/env bash
# Run on the MacBook. Pull a prepared, checksum-manifested snapshot from Lambda.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/pull_artifacts.sh SSH_HOST REMOTE_SNAPSHOT [LOCAL_BACKUP_ROOT]" >&2
  exit 2
fi
backup_host=$1
backup_source=$2
backup_root=${3:-"$HOME/preanchor-backups"}
# Reject shell metacharacters; rsync invokes a remote shell for its SSH transport.
if [[ ! $backup_host =~ ^[a-zA-Z0-9_@.-]+$ || $backup_host == -* ||
      ! $backup_source =~ ^/lambda/nfs/[a-zA-Z0-9_./-]+$ || $backup_source == *..* ]]; then
  echo "Expected an SSH host alias and an absolute /lambda/nfs/ snapshot path." >&2
  exit 2
fi
backup_name=$(basename "$backup_source")
backup_destination="$backup_root/$backup_name"
# Make a new local destination; never overwrite an earlier backup or use rsync --delete.
mkdir -p "$backup_root"
mkdir "$backup_destination"
rsync -az -- "$backup_host:$backup_source/" "$backup_destination/"
python3 - "$backup_destination" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
manifest = root / "checksums.json"
entries = json.loads(manifest.read_text())
if not entries:
    raise SystemExit("Empty backup checksum manifest")
for name, expected in entries.items():
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise SystemExit(f"Unsafe or missing backup entry: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f"Checksum mismatch: {name}")
receipt = {
    "schema_version": 1,
    "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "destination": str(root),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "verified_files": len(entries),
    "status": "secondary_copy_verified",
}
with (root / "SECONDARY_VERIFIED.json").open("x") as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2)
    stream.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
