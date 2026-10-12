# Portable backup and restore: investigation and design report

Investigation date: 2026-09-15, 15:19–15:30 UTC. Repository `codex/mvp` at `be7d855`, clean.
Read-only inspection only; no files under `PREANCHOR_ROOT` or the repository were changed.

Evidence tags: OBSERVED (code, filesystem, manifests, command output), INFERRED (derived), PROPOSED (design).

## Headline findings

1. OBSERVED: the mounted persistent filesystem is UUID `8dd29228-2370-430d-836b-a3e9de7cc148`
   (virtiofs, rw), not the `63166b57…` filesystem the docs still describe. It was empty at
   13:05 UTC today. `artifacts/reconstruction/20260915T145500Z-h100-cross-region.json` records
   a cross-region migration in which **nothing was restored from backup**
   (`restored_from_backup: []`); Stage 2 and the 20 GB raw corpus were re-downloaded, Stage 3
   and Stage 4 were recomputed, and Stage 5 benchmarks and production were restarted.
2. OBSERVED: recomputation changed scientific identity. `trajectories.parquet` embeds the
   absolute `source_path` of the Stage 2 unit (`schemas.trajectory_record`), so its hash changed
   (`07867455…` → `4fd5fbee…`), which changed the Stage 5 input hash and therefore every
   rollout ID (`CompletionSpec.rollout_id`). `labels.parquet` changed hash (`2049a33f…` →
   `fab9886e…`) because MiniLM cosine fingerprints differ between Ampere and Hopper kernels.
   Every recorded invariant matched, but byte identity did not. This is exactly the loss a
   portable backup must prevent: **the canonical Stage 3/4/5 artifacts must be restored as
   bytes, never recomputed**.
3. OBSERVED: a Stage 5 H100 production run `20260915T144926Z-bb9ba695de-9fca4c21` is live
   (124/256 rollouts at 15:19 UTC, ETA about 31 minutes). Its only checkpoint is the on-NFS
   `artifacts/backups/stage5-live-…` directory written by `pilot.checkpoint`; nothing from this
   filesystem has a secondary copy. `.env` has `PREANCHOR_BACKUP_REMOTE` empty.
4. OBSERVED: the existing backup tooling (`prepare_*_backup.py`, `pull_artifacts.sh`) covers
   Stages 0–2 only, copies whole directories into NFS snapshots, has no restore path, and is
   not content addressed. No Stage 3/4/5 prepare script exists.

---

## A. CURRENT STATE INVENTORY

Measured with `du`/`find` on `/lambda/nfs/preanchor/preanchor` (68 GB used, 1.8 PB volume).

| Path / pattern | Measured size | Stage | Canonical status | Purpose | Evidence |
|---|---|---|---|---|---|
| `data/processed/stage3/{trajectories,sentences}.parquet` + `.sha256` + `.schema.json` + `import_summary.json` | 1.25 MB (422,046 + 825,456 B parquet) | 3 | **Canonical (current)**; hashes `4fd5fbee…`, `83cf8c20…` | Exact trace import and alignment; Stage 5 input | `cli.import_data_command`, `pilot.inputs`, reconstruction record |
| `data/processed/stage4/labels.parquet` + sidecars + `labels_summary.json` | 1.5 MB | 4 | **Canonical (current)**; `fab9886e…`, 6,474 rows, 4,019 eligible, 344 anchors | Primary labels for Stages 7–10 | `stage4.build_labels(publish=True)`, `labels_summary.json` |
| `artifacts/runs/20260915T141929Z-45b6f6e0cc-07aa7070` | 1.5 MB (32 files) | 4 | Canonical audit + label-build run at `be7d855` | Audit (Spearman 0.858, gate failed → canonical fallback), labels, manifests | manifest `stages.stage4_audit/stage4_labels = complete` |
| `artifacts/runs/20260915T134433Z-d7d055d8b7-05a81618/stages/stage4_raw/units` | 20 GB (6,474 units, 45,318 files) | 4 | Reacquirable provenance run at `7357af6` | Raw fork corpus (`solutions.json` per fork) | `run_raw_fetch`; `validate-raw-resamples --no-require-identity` passed |
| same run, metadata only (`source_manifest.json` 10 MB, manifests, attempts, W&B) | 10.3 MB | 4 | Provenance referenced by hash from the label build (`raw_source_manifest`, `raw_run_manifest`) | | `stage4.load_raw_run` |
| `artifacts/runs/20260915T133816Z-d7d055d8b7-73b4c983` | 326 MB (306 units 325 MB; metadata 0.67 MB) | 2 | Reacquirable source run; Stage 3 `source_path` points into it | Exact upstream files | `run_fetch`, `import_all` |
| `artifacts/runs/20260915T132525Z-5fcd0d1c1f-9e4d25bc` | 45 MB | 2 | Failed/incidental (started from `main@6ee01ed`, stopped) | Evidence only | reconstruction `retained_incidental_runs` |
| `artifacts/runs/20260915T143802Z-…-eee71ef4` (B4), `…-4b18d0f3` (B8) | 616 KB, 543 KB | 5 | Benchmark evidence; B8 selected by `select()` | 8 completions each, `summary.json`, admission | `a6000.select`, `resource_projection.json.evidence` |
| `artifacts/runs/20260915T144926Z-bb9ba695de-9fca4c21` | 5.6 MB now (units_raw 3.8 MB for 132 units; telemetry 1.7 MB; logs 141 KB) | 5 | **Canonical production, running** | 256 rollouts, plan, admission, projection, tokenizer validation | heartbeat `state: running`, `progress.json` |
| `artifacts/backups/stage5-live-20260915T144926Z-…` | 2.5 MB, 361 files | 5 | Primary-only checkpoint copy (`secondary_verified: false`) | 15-minute live checkpoint | `pilot.checkpoint` |
| `artifacts/h100-admission/*.log` | 148 KB | 5 | Evidence; `startup.log` is also copied into each run | vLLM startup and production logs | `a6000.initialize` |
| `artifacts/reconstruction/…json` (+sha256) | 5.5 KB | — | Provenance of the migration | | inspected |
| `artifacts/doctor/*.write-probe`, `vllm-image-digest.txt` | 3 KB, 89 B | — | Admission records | | `doctor.check_persistence`, `start_vllm.sh` |
| `cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B` | 28 GB | 5–7 | Cache, pinned revision `1df85071…` | Model weights and tokenizer | refs/main |
| `cache/huggingface/hub/datasets--uzaymacar--math-rollouts` | 20 GB (6,621 blobs) | 2/4 | Cache; byte-for-byte duplicate of the raw units (blob `baf3249e…` = unit file) | HF transit cache | verified one blob by git SHA-1 |
| `cache/huggingface/datasets-hub/models--sentence-transformers--all-MiniLM-L6-v2` | 88 MB | 4/5 | Cache, revision `1110a243…` | Similarity model | refs/main |
| `cache/huggingface/xet`, `.agent_harnesses.json` | 717 KB | — | Cache | | |
| `wandb/` | 0 | — | Empty; W&B offline dirs live inside each run's `logs/` (0.03–3 MB each) | | `tracking.start_tracking(dir=run/logs)` |
| Off-filesystem: repo `.git` 131 MB, `.venv` 5.9 GB, vLLM image 28.8 GB | — | — | Regenerable | | `docker image ls`, `du` |

Metadata-only footprint of all seven runs excluding the two big unit stores: **2,191 files, 20.4 MB**.
`data/` + `backups/` + admission + reconstruction: **808 files, 7.6 MB**.

Not present on this filesystem (existed on the A6000-era filesystem per docs): `artifacts/bootstrap/*`, `artifacts/backups/stage{0,1,2}-*`, `artifacts/a6000-admission/*`, `artifacts/amendments/20260915T102700Z.md` (the user-authorized runtime amendment cited in `docs/decisions.md`), `artifacts/PRE_FIX_NON_CANONICAL_INDEX.json`, all A6000-era runs.

---

## B. FUTURE-RESUME DEPENDENCY GRAPH

Traced from consuming code, not stage order.

### Resume Stage 5 (`python -m preanchor.labels.a6000 execute|finalize --run-id R`)

| Consumer | Logical artifact | Path | Byte identity matters? | Reacquire / recompute? | Upstream |
|---|---|---|---|---|---|
| `manifest.check_resume` → `load_manifest` | run manifest current view + immutable revision | `runs/R/manifest.json`, `manifests/<sha>.json` (+ `.sha256`) | Yes (current view must equal its revision file) | No | — |
| `check_resume` | resolved config | `runs/R/resolved_config.yaml` (+sha256) | Yes; must equal `load_config(configs/pilot.yaml)` at the recorded commit | No | repo `configs/pilot.yaml`, `mvp.yaml`, `splits/mvp_20_v1.yaml` |
| `check_resume` → `provenance(repo)` | code identity | git HEAD == `be7d855`, empty dirty diff, `uv.lock` hash `1126f52a…`, submodules at `upstream.yaml` revisions | Yes | Clone | GitHub remote |
| `pilot.inputs` | Stage 3 tables | `data/processed/stage3/trajectories.parquet`, `sentences.parquet`, `.sha256`, `.schema.json` | **Yes**: their SHA-256 values are inside `input_hash` → `rollout_id` | **No** (recompute changes `source_path`, hence hash, hence all rollout IDs; observed today) | Stage 2 units (only for recompute) |
| `pilot.inputs` | pilot target mapping | repo `configs/pilot_targets_4682.yaml` (`1243a6fe…`) | Yes | Clone | — |
| `a6000.execute` | plan, tokenizer validation, admission, projection | `runs/R/stages/stage5/{plan,tokenizer_validation,runtime_admission,resource_projection_a6000}.json` (+sha256) | Yes (`read_json` validates sidecar; plan rollout IDs must equal recomputed specs) | No | benchmark runs (projection evidence paths are absolute) |
| `a6000.execute` / `finalize` → `UnitStore.validate` | completed rollouts | `runs/R/stages/stage5/units_raw/<rollout_id>/{request,response,SUCCESS}.json` + sidecars | Yes (SUCCESS lists file hashes) | No: this is the expensive GPU output | — |
| `finalize` → `CompletionStore.run_one` | scored units (written by finalize) | `runs/R/stages/stage5/units/<rollout_id>/completion.json` | Yes once written | Recomputable from `units_raw` on CPU (MiniLM) but hashes may drift across GPUs; treat as canonical once produced | `units_raw`, MiniLM cache |
| `finalize` | fresh labels | `stages/stage5/fresh_labels.parquet`, `base_validation.json` | Yes | No | above |
| `server_admission`, `tokenizer_validation` | live vLLM + startup log, tokenizer from HF cache (`local_files_only=True`) | container + `HF_HOME` model cache | No | Reacquire (pull image `sha256:61fc8a…`, model revision `1df85071…`) | HF Hub, Docker Hub |
| `select` | benchmark evidence | `runs/<bench>/stages/stage5/{summary,runtime_admission}.json`, `benchmark_units_raw/*` | Yes | Only needed to re-select; not to resume | — |

Resume also needs the run directory name unchanged (`run_path` validates charset and existence) and `PREANCHOR_ROOT` resolving under `/lambda/nfs`.

### Resume Stage 6 (extraction smoke) and Stage 7+ (SPEC §11; **not implemented**: no `model/`, `features/`, `probes/` modules exist)

INFERRED from SPEC §11 and existing schemas: Stage 6/7 consume Stage 3 `sentences.parquet` (eligible rows, `char_start` boundaries) and `trajectories.parquet` (`full_cot`), the pinned model cache, and the run manifest machinery; Stage 8+ join `labels.parquet` by `sentence_id`; Stage 9 `evaluate-test` requires `frozen_selection.json` and its hash; Stage 11 regenerates reports from `predictions/` and `metrics/` only. Future MUST-BACK-UP artifacts: activation shards (~2.9 GB, SPEC §9), `activations/index.parquet`, `features.parquet`, `probe_models/`, `predictions/`, `metrics/`, `frozen_selection.json`. No code exists yet to trace; this remains PROPOSED.

### Stage 4 (re-running audit/build, not needed to resume Stage 5)

`audit_4682`/`build_labels` read Stage 3 tables, the raw run (`load_raw_run`: manifest, `source_manifest.json`, `SUCCESS.json`, `output_hashes.stage4_raw_source_manifest`) and units by key (`raw_fork_paths` derives paths from unit keys, not stored paths), plus repo `configs/locks/raw_resample_*`. Unit keys embed the raw run's `identity` (`raw_unit_key`), so a re-downloaded corpus gets new keys and a new run ID; the label build's `input_hashes.raw_run_id` then points at a run that no longer exists. `validate_raw_download` and `validate_download` compare `row["local_path"] == str(stage/units/key/file)`: **they fail if `PREANCHOR_ROOT` is a different absolute path** (e.g. a filesystem not named `preanchor`).

### Stage 3 (re-running import)

`import_all` reads `runs/<stage2>/stages/stage2/source_manifest.json` and opens each `local_path` (absolute). Requires the Stage 2 run at the same absolute path.

---

## C. BACKUP CLASSIFICATION

| Class | Contents | Measured live size |
|---|---|---|
| **MUST BACK UP** | `data/processed/stage3/*` (1.25 MB); `data/processed/stage4/*` (1.5 MB); production run `20260915T144926Z…` excluding `telemetry/` and `progress.json`/`telemetry.json` (≈ 4.5 MB now; ≈ 8–10 MB at 256 units, more after `finalize` writes `units/` and `fresh_labels.parquet`); Stage 4 audit/build run (1.56 MB); benchmark runs B4/B8 (1.1 MB); raw-run metadata (`source_manifest.json` + `SUCCESS.json` + manifests, 10.3 MB) because the canonical labels reference them by hash; `reconstruction/*.json` (5.5 KB); `configs/locks/*`, `configs/*.yaml`, `uv.lock` (in git) | **≈ 21 MB now; ≈ 30 MB after finalize** |
| **SHOULD BACK UP** | production `telemetry/` (1.7 MB) and W&B offline dirs in `runs/*/logs` (≈ 4 MB total); `h100-admission/*.log` (148 KB); `stage5-live-*` checkpoint records (2.5 MB, mostly duplicates); Stage 2 run metadata (0.67 MB); `doctor/` probes and `vllm-image-digest.txt` | ≈ 9 MB |
| **REACQUIRE** | raw corpus units (20 GB, 6,474 files pinned by git blob SHA-1 / size in `configs/locks/raw_resample_metadata.json`; 750 s at 6 workers today); Stage 2 units (325 MB, 306 files pinned in `dataset_metadata.json`); model weights revision `1df85071…` (28 GB); MiniLM revision `1110a243…` (88 MB); vLLM image `sha256:61fc8a…` (28.8 GB) | ≈ 77 GB (of which 20.3 GB dataset, 28 GB model, 28.8 GB image) |
| **RECOMPUTE** | nothing canonical. Stage 3/4 are cheap (≈ 3 min + 11 min GPU) but **not identity-preserving**, so they are excluded from this class deliberately | 0 |
| **CACHE** | `cache/huggingface/hub/datasets--…` (20 GB, exact duplicate of the units), `xet/` (0.7 MB), `.venv` (5.9 GB), Docker image layers, `__pycache__` | 20 GB on NFS + 35 GB off-NFS |
| **HISTORICAL EVIDENCE** | failed Stage 2 run `…-9e4d25bc` (45 MB); on the old filesystem only: A6000 benchmarks/production, pre-fix runs, Stage 0–2 smoke snapshots, amendment record (not inspectable here) | 45 MB here |

---

## D. PROPOSED PORTABLE BUNDLE

PROPOSED layout on NFS at `$PREANCHOR_ROOT/artifacts/portable/`, mirrored by the MacBook at `~/preanchor-backups/portable/`:

```
portable/
  objects/sha256/<aa>/<bb>/<full-hex>          immutable blobs, name == SHA-256 of content
  manifests/<utc>-<bundle-sha>.json            bundle manifest (immutable, self-hashed)
  manifests/<utc>-<bundle-sha>.json.sha256
  latest.json                                   replaceable pointer {bundle: <name>, sha256}
  receipts/<utc>-<host>-<event>.json            prepare / pull / restore / verify receipts
```

Bundle manifest (schema_version 1):

```json
{
  "schema_version": 1, "created_at": "...", "bundle_kind": "canonical",
  "producer": {"git_sha": "...", "dirty_diff_hash": "...", "lock_hash": "...",
               "package": "preanchor 0.1.0"},
  "physical_origin": {"root": "/lambda/nfs/preanchor/preanchor",
                      "mount_source": "8dd29228-...", "fstype": "virtiofs",
                      "host": "<hostname>", "gpu": "NVIDIA H100 PCIe"},
  "scientific_identity": {
     "resolved_config_hashes": {"mvp": "c8414238...", "pilot": "9bad4b36..."},
     "scientific_config_hashes": {...},
     "revisions": {"model": "1df85071...", "tokenizer": "...", "dataset": "a3a2fe1e...",
                   "embedding": "1110a243...", "vllm_digest": "sha256:61fc8a...",
                   "vendor/thought-anchors": "b53ed8c7...", "vendor/thought-anchors.com": "67071a30..."},
     "stage_outputs": {"stage3_trajectories": "4fd5fbee...", "stage3_sentences": "83cf8c20...",
                       "stage4_labels": "fab9886e...", "stage4_labels_fingerprint": "7534066e...",
                       "stage5_production_run": "20260915T144926Z-bb9ba695de-9fca4c21",
                       "stage5_rollout_ids_sha256": "<hash of sorted plan rollout IDs>"},
     "canonical_runs": {"stage4": "20260915T141929Z-...", "stage4_raw": "20260915T134433Z-...",
                        "stage5_benchmarks": [...], "stage5_production": "..."}},
  "entries": [ {"path": "data/processed/stage3/trajectories.parquet", "sha256": "...",
                "size": 422046, "class": "must", "stage": "stage3", "run_id": null}, ... ],
  "stage_completion": {"stage5": {"state": "generated|labels_constructed", "units": 256}},
  "excluded": ["cache/**", "**/stages/stage4_raw/units/**", "**/stages/stage2/units/**", "**/progress.json", "**/telemetry.json", ".env"],
  "reacquire": [{"kind": "hf_dataset_files", "lock": "configs/locks/raw_resample_metadata.json", "bytes": 20996423633}, {"kind": "hf_model", "id": "...", "revision": "..."}, {"kind": "docker_image", "digest": "..."}]
}
```

Logical contents (paths relative to `PREANCHOR_ROOT`, plus repo pins):

- `data/processed/stage3/*`, `data/processed/stage4/*` (all files incl. `.sha256`, `.schema.json`).
- Every canonical run: `manifest.json`, `manifest.json.sha256`, `manifests/*`, `resolved_config.yaml(.sha256)`, `git.patch(.sha256)`, `environment/*`, `heartbeat.json`, `failure.json` if present, `stages/**` except large reacquirable unit stores and replaceable live files (`progress.json`, `telemetry.json`), `logs/**` (W&B offline, symlinks materialized).
- Raw-run metadata (`stages/stage4_raw/source_manifest.json`, `SUCCESS.json`, `attempts/`), not units.
- `artifacts/reconstruction/*`, `artifacts/h100-admission/*`, `artifacts/vllm-image-digest.txt`.
- Source identity: `git bundle create preanchor.bundle --all` plus the two vendor bundles (as today), stored as objects; `uv.lock` hash; `.env.example` (never `.env`).
- Receipts from previous pulls/restores.

Excluded: `cache/**`, `.venv`, Docker images, raw/Stage 2 unit payloads (reacquired by pinned locks), secrets.

Estimated size: **≈ 30 MB after Stage 5 finalize** (measured components: 20.4 MB run metadata + 7.6 MB data/evidence + ≈ 2 MB source bundles), growing to **≈ 3.0 GB after Stage 7** (SPEC estimate 2.89 GB of activations; labelled estimate).

---

## E. CURRENT BACKUP GAPS

Protected (OBSERVED from docs and receipts; the MacBook itself is not inspectable from here):

- A6000-era Stage 0 (6 files) and Stage 1 (122 files) snapshots: receipts recorded in `docs/stage0_validation.md`, `docs/stage1_validation.md`.
- Docs assert Stage 4 and "latest incremental" MacBook verification on 2026-09-14 and a plan to transfer the stopped Stage 5 run; no receipt files are in the repository, so these are **unverified claims**.

Unprotected on the current filesystem (OBSERVED: all seven manifests have `backup.verified: false`; `restored_from_backup: []`; `PREANCHOR_BACKUP_REMOTE` empty):

- Stage 3 tables `4fd5fbee…`/`83cf8c20…` and Stage 4 labels `fab9886e…` (the hashes every current rollout ID depends on).
- Stage 4 audit/build run, raw-run metadata, both benchmark runs, the reconstruction record, admission logs.
- The live production run (132 raw units so far, ≈ 1.4 GPU-hours ≈ $3.55 projected at $2.49/h). Its `stage5-live-*` checkpoint is on the same filesystem.
- Already lost from this filesystem (may exist only on the MacBook or the unreachable old filesystem): the amendment `artifacts/amendments/20260915T102700Z.md`, A6000 canonical Stage 4 labels `2049a33f…`, the stopped A6000 production run, pre-fix index.

If the filesystem disappeared now: Stage 3/4 could be recomputed (≈ 15 min plus a 20 GB download) but with **new hashes**, orphaning the production run's rollout IDs; the production run itself (all completed rollouts) would be lost and must be regenerated (≈ 1.4 GPU-hours); benchmark evidence and the reconstruction provenance would be gone.

### MacBook read-only verification commands

Run in Terminal on the MacBook. All are read-only (`ls`, `find`, `du`, `cat`, `shasum -c`, `grep`).

```bash
# 1. Which snapshots exist, with sizes and receipt status
B=~/preanchor-backups
ls -la "$B"
du -sh "$B"/*/ 2>/dev/null
for d in "$B"/*/; do printf '%s\t%s\t%s\n' "$(basename "$d")" \
  "$( [ -f "$d/SECONDARY_VERIFIED.json" ] && echo receipt || echo NO_RECEIPT )" \
  "$( [ -f "$d/checksums.json" ] && python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))),'files')" "$d/checksums.json" || echo no-manifest )"; done

# 2. Receipts and manifest digests
for f in "$B"/*/SECONDARY_VERIFIED.json; do echo "== $f"; cat "$f"; done
for f in "$B"/*/checksums.json; do printf '%s  %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$f"; done

# 3. Re-verify every snapshot's bytes against its manifest (read-only)
for d in "$B"/*/; do [ -f "$d/checksums.json" ] || continue; echo "== $(basename "$d")"
  python3 - "$d" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); bad = 0
for name, digest in json.load(open(root/"checksums.json")).items():
    p = root/name
    ok = p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
    bad += not ok
    if not ok: print("MISMATCH_OR_MISSING", name)
print("verified", "OK" if bad == 0 else f"{bad} problems")
PY
done

# 4. Does any snapshot contain the current canonical Stage 3/4/5 state (by hash or run ID)?
grep -rl -e 83cf8c20bda3ba328a3e456927b1dd22a94b584c851d10636013e3a6e9143cf8 \
        -e 4fd5fbee159c0b54d2de6c928cc48c91f7c90e4164d805de2862867c6f39913d \
        -e fab9886e478922ac132d0a98f3320c3047cb87cd6252c0d06117b99861a61b31 \
        -e 20260915T144926Z-bb9ba695de-9fca4c21 "$B" 2>/dev/null || echo "current H100-era canonical state: NOT present"

# 5. Does any snapshot hold the A6000-era canonical artifacts (useful for the labels fingerprint check)?
grep -rl -e 2049a33f6ba3fc8ab51114ad3108bd8f8340e295f53470fab0e3d4e2a163d763 \
        -e 07867455 -e 20260915T120136Z-b1a2dd9ab0-3b77595c -e 20260915T124322Z-d6cd3c11d8-5568422c \
        -e 20260915T102700Z "$B" 2>/dev/null || echo "A6000-era canonical state: NOT present"
find "$B" -name 'labels.parquet' -o -name 'trajectories.parquet' -o -name 'sentences.parquet' \
  -o -name '*.md' -path '*amendments*' 2>/dev/null | while read -r f; do printf '%s  %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$f"; done
```

Expected result today: item 4 prints "NOT present" (nothing from this filesystem has been pulled).

---

## F. FUTURE MIGRATION PROCEDURE

`current Lambda -> MacBook -> new regional filesystem -> verified resumable project`

1. **Current instance, after each stage closes** (and once the live run finishes and `finalize` runs):
   `uv run --frozen preanchor backup prepare --kind canonical` → writes new objects, a bundle manifest, `latest.json`, and a prepare receipt. Fails if the repo is dirty, any run's `manifest.json` differs from its immutable revision, any `.sha256` sidecar mismatches, any parquet fails `validate_parquet`, or a MUST entry is missing. Never touches caches or `.env`.
2. **MacBook**: `bash scripts/pull_portable_backup.sh <ssh-alias> [/lambda/nfs/preanchor/preanchor/artifacts/portable]` → rsync `manifests/` and `latest.json`, then rsync `objects/` with `--ignore-existing` (only new hashes transfer), rehash every object named by the manifest, write `receipts/<utc>-macbook-pull.json`. Fails (non-zero, no receipt) on any mismatch. The receipt is then copied back by the human into the repo docs or pulled by the agent on the next `backup verify --receipt`.
3. **New instance, empty filesystem** (name it `preanchor` so `PREANCHOR_ROOT` keeps the same absolute string; see §I):
   `git clone --recurse-submodules …` → `bash scripts/bootstrap_remote.sh --pull-vllm` → the MacBook pushes nothing; instead the human runs `bash scripts/push_portable_backup.sh <ssh-alias>` from the MacBook (rsync the `portable/` tree to the new `$PREANCHOR_ROOT/artifacts/portable/`), keeping the MacBook-initiates-transfers rule.
4. `uv run --frozen preanchor restore apply --bundle latest` → for every entry: refuse if the object is absent or rehashes differently; install with `atomic_bytes` at the logical path (refuse to overwrite an existing different file); after installation re-run `validate_artifact`/`validate_parquet`/`UnitStore.validate`/`load_manifest` on everything installed; write `receipts/<utc>-restore.json` with per-entry status `restored | already_present_identical | missing | mismatch` and the new physical identity (`findmnt` source UUID, host). Any `missing`/`mismatch` → exit non-zero and the receipt says `state: failed`; nothing is regenerated.
5. `uv run --frozen preanchor restore verify --bundle latest --for-stage 5` → runs the scientific gates of §I and writes `receipts/<utc>-verify.json` with `state: verified` only when all pass.
6. Reacquire: `preanchor fetch-raw-resamples` / `fetch` only if Stage 4 or 3 must be rerun (not for Stage 5+); model and MiniLM come through the existing `doctor`/vLLM paths.
7. Resume: `scripts/start_vllm.sh …` then `python -m preanchor.labels.a6000 execute --run-id <prod> --startup <log>` (or `finalize`). `check_resume` and `server_admission` remain the last gate.

Failure behaviour: every step is fail-closed and idempotent; a partial restore leaves only fully verified files plus a `failed` receipt, and `restore apply` can be rerun.

---

## G. TOOLING DESIGN

New package `src/preanchor/backup/` (PROPOSED), wired as a Typer sub-app in `cli.py`:

| Command | Purpose | Reuses |
|---|---|---|
| `preanchor backup plan [--kind canonical\|evidence] [--for-stage N]` | Print the entry list and sizes without writing | classification rules table (module constant), `UnitStore.validate`, `load_manifest` |
| `preanchor backup prepare` | Validate, hash, write objects + manifest + receipt on NFS | `ids.file_hash`, `io.atomic_bytes/write_json`, `manifest.provenance`, `storage.persistent_mount`, git bundle logic from `prepare_stage2_backup.py` |
| `preanchor backup verify [--bundle X] [--receipt path]` | Rehash objects on NFS; record a MacBook receipt into the run manifests' `backup.verified` | `load_manifest`, `save_manifest` (replaceable current view) |
| `preanchor restore apply --bundle X [--dry-run]` | Fail-closed installation | `atomic_bytes`, `validate_parquet`, `UnitStore.validate`, `load_manifest` |
| `preanchor restore verify --bundle X --for-stage N` | Scientific equivalence gates (§I) | `pilot.inputs`, `production_specs`, `labels_fingerprint`, `check_resume`-style identity comparison, `doctor.check_submodules` |
| `preanchor backup import-legacy <snapshot-dir>` | Ingest an old `checksums.json` snapshot into the object store | `pull_artifacts.sh` manifest format |
| `scripts/pull_portable_backup.sh`, `scripts/push_portable_backup.sh` | MacBook-side rsync + verify (bash + inline python, same style as `pull_artifacts.sh`) | `pull_artifacts.sh` guards |

`pilot.checkpoint` should write its 15-minute copies into the same object store (`objects/` + a `stage5-live` manifest) instead of `backups/stage5-live-*`, so the live checkpoint is also pullable incrementally.

---

## H. INCREMENTAL BACKUP DESIGN

Chosen: **a plain content-addressed object store plus a small path→hash manifest, transported by `rsync --ignore-existing`**.

- Object identity is the SHA-256 the project already computes everywhere (`ids.file_hash`, sidecars, `SUCCESS.json`). Unchanged artifacts map to existing object names and are never re-sent: `rsync -a --ignore-existing objects/` transfers only new hashes; the MacBook's copy is verified by rehashing objects named in the new manifest.
- New canonical outputs (e.g. `finalize` results, Stage 7 shards) are added by running `backup prepare` again: a new manifest is written, `latest.json` moves, and only new objects appear.
- Manifests are immutable and self-named by hash; deletion never happens on either side.
- Why not `--link-dest`: it dedupes by path+mtime+size across snapshot directories, is sensitive to renames (run IDs, unit keys) and mtimes rsync preserves imperfectly across virtiofs/APFS, and gives no content proof; the hash store gives verification and dedupe from the same primitive with about 200 lines of Python.
- Why not a full tool (restic/borg): extra dependency, opaque repository format, and no advantage at ≈ 30 MB–3 GB scales.
- Optional space saving on NFS: `restore apply` may hardlink or copy; copying is the default (virtiofs hardlink semantics unverified).

---

## I. VALIDATION MODEL

**Scientific identity** (what `restore verify` must prove equal):

1. Source: git SHA and empty dirty diff of the producing run (`manifest.source.git_sha`, `dirty_diff_hash`), `uv.lock` hash, submodule revisions vs `configs/locks/upstream.yaml`.
2. Config: `resolved_config_hash` and `scientific_config_hash` of the restored `resolved_config.yaml` equal both the manifest and `load_config()` at that commit.
3. Pinned revisions: model, tokenizer, dataset, MiniLM, vLLM digest (`manifest.revisions`, `manifest.vllm.digest`).
4. Inputs: `manifest.input_hashes` equal freshly computed hashes of restored inputs (`pilot.inputs` for Stage 5; `load_stage3` + `load_raw_run` metadata for Stage 4).
5. Outputs: every `.sha256` sidecar and `.schema.json` (row count, Arrow schema, data hash) validates; `manifest.output_hashes` match; `labels_fingerprint(labels.parquet)` equals the recorded fingerprint.
6. Unit integrity: `UnitStore.validate` on every restored unit; for Stage 5 the set of `units_raw` keys ⊆ plan rollout IDs, and `production_specs()` recomputed from restored inputs reproduces `plan.json` rollout IDs exactly.
7. Run identity: run IDs and manifest revision chains preserved verbatim (`load_manifest` succeeds and the current view equals its revision).
8. Runtime settings that matter scientifically: temperature 0.6, top_p 0.95, max_new_tokens 16384, seed_root 20260914, `add_special_tokens=false`, dtype bf16, `max_model_len` 20480 (frozen tuple in `a6000.plan`); concurrency/profile are execution-only and may differ.

**Physical identity** (recorded in receipts only, never compared for validity): filesystem UUID (`findmnt` source), mount target, hostname, region, GPU name, absolute `PREANCHOR_ROOT`. Restored manifests keep their original `physical_origin`; the receipt adds `physical_destination`. Absolute paths embedded in artifacts (`source_path` in `trajectories.parquet`, `local_path` in source manifests, `artifact_path`/`evidence.path` in Stage 5 plans and projections) are treated as historical strings: verification uses relative paths and hashes. Because `validate_download`/`validate_raw_download` currently compare `local_path` literally, the new filesystem should be named `preanchor` so `PREANCHOR_ROOT` is unchanged; a small backward-compatible relaxation (compare the path relative to the run directory) is listed in §J.

A restored stage is scientifically valid only when gates 1–8 pass and the receipt state is `verified`; anything `missing` is reported as such, never filled by recomputation.

---

## J. IMPLEMENTATION PLAN (no changes made)

Files to add:

- `src/preanchor/backup/__init__.py`, `classify.py` (path rules per class, one table), `manifest.py` (bundle schema, self-hash), `store.py` (object store: put/has/verify, `atomic_bytes`-based), `prepare.py`, `restore.py`, `verify.py` (gates of §I), `receipts.py`.
- `src/preanchor/cli.py`: `backup` and `restore` Typer sub-apps.
- `scripts/pull_portable_backup.sh`, `scripts/push_portable_backup.sh` (MacBook side; same input guards as `pull_artifacts.sh`).
- `docs/backup_restore.md` (procedure, receipts, MacBook commands); update `docs/lambda_runbook.md`, `docs/continuation.md` (record the reconstruction and the new filesystem UUID; the docs still cite the A6000 filesystem and run IDs), `.env.example` (`PREANCHOR_BACKUP_REMOTE` semantics).
- `tests/unit/test_backup_classify.py`, `tests/unit/test_backup_manifest.py`, `tests/integration/test_backup_roundtrip.py`, `tests/integration/test_restore_fail_closed.py`, `tests/integration/test_pull_portable_backup.py` (bash script with fake rsync, as in `test_bootstrap_backup.py`).

Files to modify:

- `src/preanchor/labels/pilot.py` `checkpoint()`: write into the object store (keep the return contract used by `a6000.execute`/`finalize`).
- `src/preanchor/data/stage.py` `validate_download` and `src/preanchor/labels/stage4.py` `validate_raw_download`: compare `local_path` relative to the run directory (accept legacy absolute equality), with a regression test; needed for restored runs under a different `PREANCHOR_ROOT`.
- `src/preanchor/data/schemas.py` `trajectory_record.source_path`: leave as is for now (changing it changes Stage 3 hashes and rollout IDs); record an amendment proposal for the next identity boundary.
- `docs/decisions.md`: entries for the backup design and the path-comparison relaxation.

Existing machinery: keep `prepare_bootstrap_backup.py`, `prepare_stage1_backup.py`, `prepare_stage2_backup.py`, `pull_artifacts.sh` as legacy (importable via `backup import-legacy`); replace their role for Stage 3+ with `backup prepare`.

Tests required: classification of every live path class; manifest self-hash and tamper detection; round trip on `tmp_path` using existing fixtures (create a smoke run through `run_smoke`, Stage 2 fake download, and a `FakeBackend` Stage 5 unit; prepare → wipe → restore into a second root → verify passes); fail-closed cases (missing object, corrupted object, manifest current-view mismatch, extra unrecorded file, sidecar mismatch, dirty repo) each leaving no partial install and a `failed` receipt; relocation (different `PREANCHOR_ROOT`) with the relaxed path check; idempotent re-prepare adds zero objects when nothing changed; `checkpoint` round trip.

End-to-end restore test on hardware: after the live run finalizes, prepare the bundle, pull to the MacBook, restore into a scratch root on this instance (`/lambda/nfs/preanchor/restore-test`, not the canonical root), run `restore verify --for-stage 5`, and confirm `check_resume` + `plan()` reproduce the production rollout IDs; then remove only the scratch root with explicit authorization.

Migration/compatibility: bundle schema versioned; legacy `checksums.json` snapshots importable; receipts never overwrite; nothing under `PREANCHOR_ROOT` is deleted.

---

## Immediate recommendations (no action taken)

1. Let the production run finish and run `finalize`; do not restart or touch it.
2. Approve the tooling above; the first `backup prepare` + MacBook pull would protect the ≈ 30 MB canonical state within minutes and end the current zero-secondary-copy condition.
3. Run the MacBook commands in §E to learn whether the A6000-era labels `2049a33f…`, the amendment record, and the stopped A6000 run survive anywhere; if they do, `backup import-legacy` should absorb them.
