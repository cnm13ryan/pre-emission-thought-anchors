# Backup and restore SOP (portable, content-addressed)

Routine commands first. Everything below the first section is reference.

## Routine

On the MacBook, after every canonical stage completes (and any time before a Lambda
filesystem might be lost):

```bash
./scripts/preanchor_portable.sh backup <ssh-host> --for-stage 5      # plan, prepare, pull, verify
./scripts/preanchor_portable.sh list                                  # what is verified locally
./scripts/preanchor_portable.sh verify                                # rehash the local copy
```

Verified real-world run (2026-09-15, first canonical secondary copy of the finalized H100
Stage 5 state; SSH alias `preanchor-h100` in `~/.ssh/config` on the MacBook):

```bash
PREANCHOR_REMOTE_CMD='/home/ubuntu/.local/bin/uv run --frozen preanchor' \
./scripts/preanchor_portable.sh backup preanchor-h100 --for-stage 5
```

`PREANCHOR_REMOTE_CMD` is needed when non-interactive SSH cannot resolve `uv`: the
remote command runs in a non-login shell whose `PATH` lacks `~/.local/bin`, where the
bootstrap installs `uv`. Result: bundle
`20260915T163333Z-d565605b95187a71851608d5dd1f31ad94eaffe3ae29674084da1462e83132a2`, 4,282 objects / 33,704,252 bytes rehashed on the MacBook, 0 missing, 0 corrupt,
receipt `~/preanchor-backups/portable/receipts/20260915T163431Z-pull_verification-d565605b9518-62c783c6.json` (copied back to
`artifacts/portable/receipts/` on Lambda), final line `BACKUP VERIFIED`.

Stage 6 (2026-09-15): bundle
`20260915T180804Z-f4cd4b0b8f03569ba33b87502010f3d015214a3157469b56d947faf1d73e6d7d` was
prepared on Lambda (152 new objects, 216 MB beyond the Stage 5 store), restore-verified
into a scratch root, and then independently verified on the MacBook (4,434 objects,
250,021,342 bytes, 0 missing, 0 corrupt, receipt
`~/preanchor-backups/portable/receipts/20260915T181939Z-pull_verification-f4cd4b0b8f03-d185f734.json`, copied back to
Lambda, final line `BACKUP VERIFIED`). The Stage 5 secondary copy above stays as it is. The MacBook pull for
Stage 6 must run against a checkout that contains Stage 6 backup support (`128e1d3` or a
descendant); `~/preanchor-repo` is still on `codex/mvp`, so pass `PREANCHOR_REMOTE_REPO`:

```bash
PREANCHOR_REMOTE_REPO=/home/ubuntu/preanchor-stage6 PREANCHOR_REMOTE_CMD='/home/ubuntu/.local/bin/uv run --frozen preanchor' ./scripts/preanchor_portable.sh backup preanchor-h100 --for-stage 6 --no-prepare
```

`--no-prepare` pulls the already prepared and restore-verified bundle (`latest`); without
it Lambda prepares a fresh bundle with identical content and a new ID first. Either way
the Stage 6 secondary copy is verified only when the command ends with `BACKUP VERIFIED`
and a `pull_verification` receipt exists. What a `--for-stage 6` bundle adds: every
complete Stage 6 run matching the current Stage 3/4 tables (activation shards and unit
records, `activations/index.parquet`, the Stage 6 summary, invariance report, resource
projection, boundary audit and logit features, run manifests and configs); Stages 3–5 are
required dependencies of the bundle.

Stage 7 (2026-09-15): the canonical Stage 7 run (`20260915T182306Z-3ab408d80a-7452f1c2`,
40 activation units, 6,474 rows, 3.31 GB of float16 shards, the activation index, the
extraction summary and projection, and the feature table with its audit and validation
record) plus the published `data/processed/stage7/{features.parquet, .schema.json,
feature_audit.json}` and their `.sha256` sidecars are protected by
`--for-stage 7` (`src/preanchor/backup/classify.py::discover_stage7`); Stages 3–6 are its
required dependencies (`STAGE_DEPENDENCIES[7] = (3, 4, 5, 6)`, Stage 6 being the
measured-fit admission evidence). `data/processed/stage7/superseded/` (relocated smoke
publications) is historical. `restore verify --for-stage 7` re-runs `validate-activations`
and a read-only `validate-features` on the restored bytes and checks the index, shard and
feature hashes, feature columns, head weights, frozen extraction settings and the
published-copy identity against the bundle identity. The bundle IDs and the real scratch
round trip are in `docs/stage7_validation.md`. The MacBook pull must run against a checkout
that contains Stage 7 backup support (`56aea1b` or a descendant on `codex/stage7`):

```bash
PREANCHOR_REMOTE_REPO=/home/ubuntu/preanchor-stage7 PREANCHOR_REMOTE_CMD='/home/ubuntu/.local/bin/uv run --frozen preanchor' ./scripts/preanchor_portable.sh backup preanchor-h100 --for-stage 7 --no-prepare
```

**Stages 8–10.** `--for-stage 8` protects the canonical Stage 8 run (`discover_stage8`:
finalized fit, selection, validated probes on the canonical Stage 7 inputs), `--for-stage 9`
the frozen, test-evaluated Stage 9 run on that selection (`discover_stage9`), and
`--for-stage 10` the complete Stage 10 run whose `input_hashes` are exactly the canonical
Stage 9 run, its freeze hash and every Stage 9 output hash (`discover_stage10`;
`STAGE_DEPENDENCIES[10] = (3, …, 9)`). Stage 10 protected bytes: the three tables
(`metrics/sensitivities.parquet`, `metrics/permutations.parquet`,
`predictions/lead_time.parquet`), the figures (html/json with sidecars), the 22 immutable
units under `stages/stage10/units/` and the summary record. An interrupted Stage 10 run and
a Stage 10 run on other inputs are historical. `restore verify --for-stage 10` re-runs
`validate-sensitivities` read-only on the restored run (Stage 9 inputs unchanged,
train-only thresholds with q90 equal to the freeze, frozen-site-only models, lead-one
mapping, permutation count/seeds/strata). The MacBook pull must run against a checkout
that contains Stage 10 backup support (the Stage 10 worker branch or a descendant):

```bash
PREANCHOR_REMOTE_REPO=/home/ubuntu/preanchor-stage10 PREANCHOR_REMOTE_CMD='/home/ubuntu/.local/bin/uv run --frozen preanchor' ./scripts/preanchor_portable.sh backup preanchor-h100 --for-stage 10
```

Before terminating a Lambda instance whose persistent filesystem you may never mount
again: run the `backup` command above and require the literal line `BACKUP VERIFIED` and a
`pull_verification` receipt under `~/preanchor-backups/portable/receipts/`. Nothing else
counts. A bundle prepared on Lambda (`preanchor backup prepare`) is primary-only.

Onto a fresh persistent filesystem in a new region (new UUID, host, GPU, mount path):

```bash
# new Lambda instance
git clone <repo> ~/preanchor-repo && cd ~/preanchor-repo && git checkout <producing commit>
# recreate .env by hand (secrets are never in a bundle); set PREANCHOR_ROOT and the hourly rate
bash scripts/bootstrap_remote.sh

# MacBook
./scripts/preanchor_portable.sh restore <new-ssh-host> --for-stage 5
```

The restore ends with `RESTORE VERIFIED`, `Scientifically equivalent portable state: PASS`
and `Ready to continue Stage 5: YES|NO`. `NO` means the checkout is not the producing
commit (or is dirty / has another `uv.lock`): check out that commit and rerun
`uv run --frozen preanchor restore verify --bundle <id> --for-stage 5`.

Lambda-side equivalents (all read-only except `prepare`/`apply`):

```bash
uv run --frozen preanchor backup plan --for-stage 5
uv run --frozen preanchor backup prepare --for-stage 5
uv run --frozen preanchor backup verify --bundle latest
uv run --frozen preanchor backup list
uv run --frozen preanchor backup import-legacy artifacts/backups/<legacy-snapshot>
uv run --frozen preanchor restore plan  --bundle <id> --for-stage 5
uv run --frozen preanchor restore apply --bundle <id> --for-stage 5 [--dry-run]
uv run --frozen preanchor restore verify --bundle <id> --for-stage 5
```

`--root` overrides `PREANCHOR_ROOT` (used for scratch restores such as
`/lambda/nfs/preanchor/restore-test`); `--store` points at another portable store.

## What is backed up

The store lives at `$PREANCHOR_ROOT/artifacts/portable/` and the MacBook mirror at
`~/preanchor-backups/portable/`:

```text
objects/sha256/<aa>/<bb>/<sha256>     one immutable payload per distinct content
manifests/<UTC>-<manifest-sha256>.json (+ .sha256)   immutable bundle manifests
receipts/<UTC>-<kind>-<sha12>-<rand>.json            append-only receipts
latest.json                           replaceable pointer to the newest manifest
```

Classification (`preanchor.backup.classify`) is derived from manifests and validation
records on the filesystem, never from run IDs in code:

| Class | Content | Bundled |
| --- | --- | --- |
| `MUST_BACK_UP` | `data/processed/stage3/**`, `data/processed/stage4/**`, the Stage 4 build run (labels, audit, summaries, manifests), Stage 4 raw-run provenance (manifest revisions, `source_manifest.json`, `SUCCESS.json`, attempts, config, patch) without its units, the Stage 5 production run (manifests, resolved config, patch, plan, tokenizer validation, runtime admission, projections, doctor, summary, `units_raw/**`, `units/**`, base validation, fresh labels) and the benchmark runs its projection cites, the complete Stage 6 runs on the current Stage 3/4 tables (`activations/**`, `stages/stage6/**`), the canonical Stage 7 run (`activations/**`, `features/**`, `stages/stage7/**`) and the published `data/processed/stage7/{features.parquet, .schema.json, feature_audit.json}` with sidecars, `artifacts/reconstruction/**`, `artifacts/amendments/**` | yes, exact bytes |
| `SHOULD_BACK_UP` | heartbeat, progress, telemetry, startup and admission logs, W&B offline files, environment captures, doctor evidence, live-checkpoint indexes, Stage 2 run metadata | yes |
| `REACQUIRE` | Stage 2 dataset payload, Stage 4 raw resample payload, DeepSeek weights and tokenizer, MiniLM, the vLLM image | no; the pin (revision, digest, source-manifest hash) is recorded in the manifest |
| `CACHE_EXCLUDE` | `cache/**`, `.venv`, Docker layers, `.locks`, `quarantine`, partial units, `.run.lock`, `artifacts/portable` itself, primary-only `stage5-live-*/run` copies | no |
| `HISTORICAL_EVIDENCE` | failed, superseded and non-dependency runs, legacy snapshots, `data/processed/stage7/superseded/` | only with `--include-historical` or `import-legacy` |
| `SECRET` | `.env*`, credentials, keys, any file whose content looks like a token | never; a secret inside a run blocks `prepare` |

Stage dependencies used for `--for-stage`: Stage 4 consumes Stage 3; Stage 5 consumes
Stage 3 and `configs/pilot_targets_4682.yaml` (its rollout IDs hash both); Stage 6 and
Stage 7 consume Stage 3 and the Stage 4 labels (`features.extract.load_inputs`), Stage 7
also carrying Stage 5 (continuation) and Stage 6 (admission evidence). A bundle for
stage N carries every completed canonical stage up to N; the transitive dependencies are
flagged `required`.

## Why exact bytes

Canonical derived artifacts are not byte-reproducible: Stage 3 `trajectories.parquet`
embeds the absolute Stage 2 `source_path`, Stage 4 MiniLM cosine scores differ between
GPU generations, and the Stage 3 hashes feed `manifest.input_hashes` and every Stage 5
rollout ID. The 2026-09-15 cross-region reconstruction reproduced equivalent science with
different hashes and therefore a different production plan. Restoring exact bytes is the
only way to continue a run rather than restart it.

## Incremental behaviour

Identity is SHA-256, never mtime. Preparing the same state twice adds zero objects and one
new manifest; new completions add exactly their files. The MacBook pulls only objects it
does not have (`rsync -a --ignore-existing --files-from=<manifest object list>`), then
rehashes every object the manifest references and checks the manifest against its own
name and sidecar. Only then does it write the `pull_verification` receipt and print
`BACKUP VERIFIED`; the receipt is copied back to Lambda over the same Mac-initiated
connection. Old objects and manifests are never deleted by the tooling.

## Live Stage 5 checkpoints

`backup prepare` on a running production run produces a `live_checkpoint` bundle:
only units with a valid `SUCCESS.json` (checked with `UnitStore.validate`) are ingested,
partial and corrupt units are excluded, and the manifest records
`stage5.status: running`, `protected_completed_units`, `resumable` and
`finalized: false`. `highest_protected_stage` stays below 5. The existing 15-minute
primary checkpoint (`pilot.checkpoint`) is unchanged. After `finalize`, a normal
`backup` produces the canonical bundle; the units already protected are reused.

## Scientific versus physical identity

`scientific_identity` in the manifest holds resolved/scientific config hashes, producing
commits, dirty-diff and `uv.lock` hashes, submodule revisions, model/tokenizer/dataset/
MiniLM revisions, the vLLM digest, Stage 3 hashes, Stage 4 label hash and fingerprint,
run IDs, the Stage 5 input hashes, plan hash, sorted rollout-ID-set hash and the frozen
generation settings. `physical_origin` (root path, mount source, filesystem type, host,
GPU, region) is provenance only; `restore verify` prints origin and destination and never
compares them. Run IDs, manifest revision chains and the recorded origin are preserved on
restore; the new destination is written to the restore receipts.

`restore verify --for-stage 5` recomputes `pilot.inputs` on the restored Stage 3 tables,
compares the hashes with `manifest.input_hashes`, recomputes `production_specs()` and
requires its sorted rollout IDs to equal the plan exactly, validates every unit with
`UnitStore.validate`, checks `units_raw` against the plan, and checks the frozen settings
(temperature 0.6, top_p 0.95, max_new_tokens 16384, seed_root 20260914,
add_special_tokens false, dtype bfloat16, max_model_len 20480, no chat template, no
quantization) plus the pinned revisions and vLLM digest.

## Reacquiring caches

Nothing reacquirable is needed for identity. To continue work:

```bash
scripts/start_vllm.sh configs/lambda/<profile>.yaml     # pulls the pinned image by digest
uv run --frozen preanchor fetch --run-id ...             # Stage 2 payload (pinned dataset rev)
uv run --frozen preanchor fetch-raw-resamples ...        # Stage 4 raw payload; validate with
uv run --frozen preanchor validate-raw-resamples --no-require-identity
```

Model, tokenizer and MiniLM snapshots download on first use at the pinned revisions. Each
manifest's `reacquire` list names the resource, pin and source of the pin.

## Failure handling

* `backup plan` exits 2 with `BLOCKERS` when required state is absent, a checksum sidecar
  or Parquet schema mismatches, a run manifest's current view differs from its immutable
  revision, a unit is corrupt or outside the plan, or a credential is found.
* `backup prepare` refuses to run with blockers, never regenerates missing artifacts,
  never overwrites a different object under the same name, and never deletes.
* `restore apply` records per entry `restored`, `already_present_identical`, `missing`,
  `mismatch`, `conflict` (destination holds different bytes; never overwritten) or
  `validation_failed`, writes a `restore` receipt with `outcome: failed` and exits 1 on any
  non-success. Completed entries stay in place; rerunning is idempotent.
* `restore verify` fails on any changed byte in a required entry, a changed Stage 3 hash,
  a rollout-ID difference or a frozen-setting difference.
* The MacBook script exits non-zero on any rsync, hash or manifest error and never prints
  `BACKUP VERIFIED` in that case.

## Legacy snapshots

`scripts/prepare_bootstrap_backup.py`, `prepare_stage1_backup.py`,
`prepare_stage2_backup.py` and `scripts/pull_artifacts.sh` keep working unchanged.
`preanchor backup import-legacy <snapshot-dir>` rehashes every `checksums.json` entry
first and only then stores the payloads as a `legacy_import` bundle under
`legacy/<snapshot>/...`. Whether an older MacBook snapshot exists is established only by
inspecting `~/preanchor-backups/` on the MacBook or by a receipt, never assumed.

## Listing and inspecting

`preanchor backup list` shows the store, the `latest.json` pointer, every manifest (kind,
stage, protected stage, entry count, logical bytes) and receipts. Manifests are plain
canonical JSON; `preanchor backup verify --bundle <id>` rehashes their objects.
