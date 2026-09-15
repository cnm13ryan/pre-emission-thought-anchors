#!/usr/bin/env bash
# MacBook-side portable backup/restore driver. Runs on macOS or Linux with bash 3.2+,
# ssh, rsync (macOS 2.6.9 flags only) and python3. The MacBook initiates every transfer;
# Lambda never connects to the MacBook.
#
#   ./scripts/preanchor_portable.sh backup  <ssh-host> [--for-stage N] [--dry-run] [--no-prepare]
#   ./scripts/preanchor_portable.sh verify  [--bundle ID]
#   ./scripts/preanchor_portable.sh list
#   ./scripts/preanchor_portable.sh restore <ssh-host> [--for-stage N] [--bundle ID] [--dry-run]
#
# Environment: PREANCHOR_BACKUP_DIR (default ~/preanchor-backups/portable),
#              PREANCHOR_REMOTE_REPO (default ~/preanchor-repo),
#              PREANCHOR_REMOTE_CMD (default "uv run --frozen preanchor"),
#              PREANCHOR_SSH / PREANCHOR_RSYNC (test overrides for the ssh/rsync binaries).
set -euo pipefail

BACKUP_DIR=${PREANCHOR_BACKUP_DIR:-"$HOME/preanchor-backups/portable"}
REMOTE_REPO=${PREANCHOR_REMOTE_REPO:-'~/preanchor-repo'}
REMOTE_CMD=${PREANCHOR_REMOTE_CMD:-"uv run --frozen preanchor"}
SSH=${PREANCHOR_SSH:-ssh}
RSYNC=${PREANCHOR_RSYNC:-rsync}

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
die() { echo "error: $*" >&2; exit 1; }

check_host() {
  # ssh/rsync spawn a remote shell; refuse anything but a plain host alias.
  [[ $1 =~ ^[a-zA-Z0-9_@.-]+$ && $1 != -* ]] || die "expected a plain SSH host alias, got '$1'"
}

remote() {
  # Remote commands source the repo's .env for PREANCHOR_ROOT; secrets never leave Lambda.
  local host=$1; shift
  "$SSH" "$host" "cd $REMOTE_REPO && set -a && . ./.env && set +a && $REMOTE_CMD $*"
}

# The restore push mutates the Lambda-side portable store (objects and manifests, never
# latest.json), so it holds portable-store:writer on Lambda for the duration of the push
# (docs/concurrency.md). The lease is released on every exit path, success or failure.
RESTORE_HOST= RESTORE_SESSION= RESTORE_TOKEN=
restore_cleanup() {
  local status=$?
  if [[ -n $RESTORE_TOKEN ]]; then
    remote "$RESTORE_HOST" "lease release portable-store:writer --token $RESTORE_TOKEN --reason restore-push-end" >/dev/null \
      || echo "warning: could not release portable-store:writer on $RESTORE_HOST; reclaim it there" >&2
    RESTORE_TOKEN=
  fi
  if [[ -n $RESTORE_SESSION ]]; then
    remote "$RESTORE_HOST" "agent close --session $RESTORE_SESSION --outcome restore-push-end" >/dev/null || true
    RESTORE_SESSION=
  fi
  return $status
}

json_field() {  # json_field <file> <dotted.key>
  python3 - "$1" "$2" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print(value if not isinstance(value, (dict, list)) else json.dumps(value))
PY
}

verify_local() {  # verify_local <bundle-id|latest> ; writes an immutable pull receipt
  python3 - "$BACKUP_DIR" "$1" "${2:-}" "${3:-}" <<'PY'
import datetime, hashlib, json, os, pathlib, re, sys, uuid

store = pathlib.Path(sys.argv[1]).resolve()
selector, host, remote_store = sys.argv[2], sys.argv[3], sys.argv[4]
name_re = re.compile(r"^(\d{8}T\d{6}Z)-([0-9a-f]{64})$")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if selector == "latest":
    selector = json.loads((store / "latest.json").read_bytes())["bundle_id"]
match = name_re.match(selector)
if not match:
    raise SystemExit(f"malformed bundle id: {selector}")
manifest_path = store / "manifests" / f"{selector}.json"
payload = manifest_path.read_bytes()
digest = hashlib.sha256(payload).hexdigest()
sidecar = (store / "manifests" / f"{selector}.json.sha256").read_text().strip()
if digest != match.group(2) or sidecar != digest:
    raise SystemExit("manifest hash does not match its name/sidecar (tampered or partial)")
manifest = json.loads(payload)
if manifest.get("schema_version") != 1:
    raise SystemExit("unsupported bundle schema")
objects = {}
for entry in manifest["entries"]:
    objects.setdefault(entry["sha256"], entry["size"])
missing, corrupt, verified_bytes = [], [], 0
for d in sorted(objects):
    path = store / "objects" / "sha256" / d[:2] / d[2:4] / d
    if path.is_symlink() or not path.is_file():
        missing.append(d); continue
    if sha256(path) != d:
        corrupt.append(d); continue
    verified_bytes += path.stat().st_size
ok = not missing and not corrupt
receipt = {
    "schema_version": 1, "kind": "pull_verification",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "bundle_id": selector, "manifest_sha256": digest,
    "run_ids": manifest.get("scientific_identity", {}).get("run_ids", []),
    "source": f"{host}:{remote_store}" if host else None, "destination": str(store),
    "object_count": len(objects), "verified_bytes": verified_bytes,
    "missing": missing, "corrupt": corrupt,
    "outcome": "secondary_copy_verified" if ok else "failed",
    "highest_protected_stage": manifest["bundle"].get("highest_protected_stage"),
    "kind_of_bundle": manifest["bundle"].get("kind"),
    "stage_status": {k: v.get("status") for k, v in manifest["bundle"].get("stages", {}).items()},
}
receipts = store / "receipts"
receipts.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
path = receipts / f"{stamp}-pull_verification-{digest[:12]}-{uuid.uuid4().hex[:8]}.json"
body = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
with path.open("xb") as stream:
    stream.write(body); stream.flush(); os.fsync(stream.fileno())
with pathlib.Path(str(path) + ".sha256").open("x") as stream:
    stream.write(hashlib.sha256(body).hexdigest() + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
print(f"RECEIPT {path}")
if not ok:
    raise SystemExit(f"verification FAILED: {len(missing)} missing, {len(corrupt)} corrupt")
PY
}

object_list() {  # object_list <bundle-id> -> relative object paths of a manifest, one per line
  python3 - "$BACKUP_DIR" "$1" <<'PY'
import json, pathlib, sys
store, name = pathlib.Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((store / "manifests" / f"{name}.json").read_bytes())
for d in sorted({e["sha256"] for e in manifest["entries"]}):
    print(f"objects/sha256/{d[:2]}/{d[2:4]}/{d}")
PY
}

cmd_backup() {
  local host=$1; shift
  check_host "$host"
  local stage=5 dry_run=0 do_prepare=1
  while [[ $# -gt 0 ]]; do
    case $1 in
      --for-stage) stage=$2; shift 2 ;;
      --dry-run) dry_run=1; shift ;;
      --no-prepare) do_prepare=0; shift ;;
      *) usage ;;
    esac
  done
  [[ $stage =~ ^[0-9]+$ ]] || die "--for-stage must be an integer"
  mkdir -p "$BACKUP_DIR/manifests" "$BACKUP_DIR/objects" "$BACKUP_DIR/receipts"
  local work; work=$(mktemp -d "${TMPDIR:-/tmp}/preanchor-portable.XXXXXX")
  echo "== remote backup plan (stage $stage)"
  remote "$host" "backup plan --for-stage $stage --json" > "$work/plan.json" || {
    cat "$work/plan.json" >&2; die "remote backup plan reported blockers"; }
  python3 - "$work/plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
c = plan["classification"]; o = plan["objects"]
print(f"kind={plan['kind']} highest_protected_stage={plan['highest_protected_stage']}")
for stage, info in plan["stages"].items():
    print(f"  {stage}: {info.get('status')}")
print(f"MUST {c['MUST_BACK_UP']['files']} files/{c['MUST_BACK_UP']['bytes']} B, "
      f"SHOULD {c['SHOULD_BACK_UP']['files']} files/{c['SHOULD_BACK_UP']['bytes']} B, "
      f"REACQUIRE {c['REACQUIRE']['bytes']} B excluded")
print(f"objects: {o['unique']} unique, {o['new']} new on Lambda ({o['new_bytes']} B)")
PY
  local remote_store; remote_store=$(json_field "$work/plan.json" objects.store)
  if [[ $dry_run == 1 ]]; then
    echo "dry run: no bundle prepared, nothing transferred"; rm -rf "$work"; return 0
  fi
  local bundle
  if [[ $do_prepare == 1 ]]; then
    echo "== remote backup prepare"
    remote "$host" "backup prepare --for-stage $stage --json" > "$work/prepare.json" || {
      cat "$work/prepare.json" >&2; die "remote backup prepare failed"; }
    bundle=$(json_field "$work/prepare.json" bundle_id)
    echo "prepared $bundle ($(json_field "$work/prepare.json" objects_created) new objects on Lambda)"
  else
    remote "$host" "backup list --json" > "$work/list.json"
    bundle=$(json_field "$work/list.json" latest.bundle_id)
  fi
  echo "== fetch immutable manifest $bundle"
  "$RSYNC" -a -- "$host:$remote_store/manifests/$bundle.json" \
    "$host:$remote_store/manifests/$bundle.json.sha256" "$BACKUP_DIR/manifests/"
  "$RSYNC" -a -- "$host:$remote_store/latest.json" "$BACKUP_DIR/latest.json"
  echo "== rsync missing content-addressed objects"
  object_list "$bundle" > "$work/objects.txt"
  local before; before=$(find "$BACKUP_DIR/objects" -type f | wc -l | tr -d ' ')
  # --ignore-existing: object names are hashes, so an existing file is never re-sent;
  # correctness comes from the independent rehash below, not from rsync.
  "$RSYNC" -a --ignore-existing --files-from="$work/objects.txt" -- \
    "$host:$remote_store/" "$BACKUP_DIR/"
  local after; after=$(find "$BACKUP_DIR/objects" -type f | wc -l | tr -d ' ')
  echo "transferred $((after - before)) new objects ($(wc -l < "$work/objects.txt" | tr -d ' ') referenced)"
  echo "== independent local verification"
  verify_local "$bundle" "$host" "$remote_store" > "$work/verify.txt" || {
    cat "$work/verify.txt"; die "local verification failed; NOT a verified secondary copy"; }
  cat "$work/verify.txt"
  local receipt; receipt=$(sed -n 's/^RECEIPT //p' "$work/verify.txt")
  echo "== copy verification receipt back to Lambda (Mac-initiated)"
  "$RSYNC" -a --ignore-existing -- "$receipt" "$receipt.sha256" "$host:$remote_store/receipts/" \
    || echo "warning: receipt copy-back failed; the local receipt is authoritative"
  rm -rf "$work"
  echo "BACKUP VERIFIED"
  echo "bundle $bundle independently verified in $BACKUP_DIR"
}

cmd_verify() {
  local bundle=latest
  while [[ $# -gt 0 ]]; do
    case $1 in --bundle) bundle=$2; shift 2 ;; *) usage ;; esac
  done
  verify_local "$bundle" "" "" && echo "BACKUP VERIFIED"
}

cmd_list() {
  python3 - "$BACKUP_DIR" <<'PY'
import json, pathlib, re, sys
store = pathlib.Path(sys.argv[1])
print(f"store: {store}")
latest = store / "latest.json"
print("latest:", json.loads(latest.read_bytes())["bundle_id"] if latest.is_file() else None)
for path in sorted((store / "manifests").glob("*.json")) if (store / "manifests").is_dir() else []:
    if not re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{64}$", path.stem):
        continue
    m = json.loads(path.read_bytes())["bundle"]
    print(f"  {path.stem}  kind={m['kind']} for_stage={m['for_stage']} "
          f"protected={m['highest_protected_stage']} "
          f"stages={ {k: v.get('status') for k, v in m.get('stages', {}).items()} }")
verified = {}
for path in sorted((store / "receipts").glob("*.json")) if (store / "receipts").is_dir() else []:
    r = json.loads(path.read_bytes())
    if r.get("kind") == "pull_verification" and r.get("outcome") == "secondary_copy_verified":
        verified[r["bundle_id"]] = r["created_at"]
print("verified secondary copies:")
for bundle, when in verified.items():
    print(f"  {bundle}  verified_at={when}")
if not verified:
    print("  (none)")
PY
}

cmd_restore() {
  local host=$1; shift
  check_host "$host"
  local stage=5 bundle=latest dry_run=0
  while [[ $# -gt 0 ]]; do
    case $1 in
      --for-stage) stage=$2; shift 2 ;;
      --bundle) bundle=$2; shift 2 ;;
      --dry-run) dry_run=1; shift ;;
      *) usage ;;
    esac
  done
  [[ $stage =~ ^[0-9]+$ ]] || die "--for-stage must be an integer"
  echo "== verify local copy before sending anything"
  local work; work=$(mktemp -d "${TMPDIR:-/tmp}/preanchor-portable.XXXXXX")
  verify_local "$bundle" "" "" > "$work/verify.txt" || { cat "$work/verify.txt"; die "local bundle failed verification"; }
  if [[ $bundle == latest ]]; then bundle=$(json_field "$BACKUP_DIR/latest.json" bundle_id); fi
  echo "== remote store location"
  remote "$host" "backup store-path --json" > "$work/store.json"
  local remote_store; remote_store=$(json_field "$work/store.json" store)
  echo "== lease portable-store:writer on $host for the push"
  remote "$host" "agent register --task restore-push-$bundle --role backup-restore --worktree . --json" \
    > "$work/session.json" || die "could not register the restore session on $host"
  RESTORE_HOST=$host
  RESTORE_SESSION=$(json_field "$work/session.json" session_id)
  trap restore_cleanup EXIT
  if ! remote "$host" "lease acquire portable-store:writer --session $RESTORE_SESSION --purpose restore-push --json" \
      > "$work/lease.json"; then
    cat "$work/lease.json"
    die "portable store on $host is held by another writer; nothing was pushed (retry later)"
  fi
  RESTORE_TOKEN=$(json_field "$work/lease.json" owner_token)
  echo "== push manifest and required objects to $host:$remote_store (Mac-initiated)"
  object_list "$bundle" > "$work/objects.txt"
  "$RSYNC" -a --ignore-existing --files-from="$work/objects.txt" -- "$BACKUP_DIR/" "$host:$remote_store/"
  "$RSYNC" -a --ignore-existing -- "$BACKUP_DIR/manifests/$bundle.json" \
    "$BACKUP_DIR/manifests/$bundle.json.sha256" "$host:$remote_store/manifests/"
  echo "== remote object verification"
  remote "$host" "backup verify --bundle $bundle" || die "remote store verification failed"
  restore_cleanup   # the store is consistent again: release before the (root-side) apply
  if [[ $dry_run == 1 ]]; then
    remote "$host" "restore apply --bundle $bundle --for-stage $stage --dry-run"
    rm -rf "$work"; return 0
  fi
  echo "== remote restore apply"
  remote "$host" "restore apply --bundle $bundle --for-stage $stage" || die "restore apply failed (fail-closed)"
  echo "== remote restore verify"
  remote "$host" "restore verify --bundle $bundle --for-stage $stage" || die "restore verification failed"
  rm -rf "$work"
}

[[ $# -ge 1 ]] || usage
command=$1; shift
case $command in
  backup)  [[ $# -ge 1 ]] || usage; cmd_backup "$@" ;;
  verify)  cmd_verify "$@" ;;
  list)    cmd_list "$@" ;;
  restore) [[ $# -ge 1 ]] || usage; cmd_restore "$@" ;;
  *) usage ;;
esac
