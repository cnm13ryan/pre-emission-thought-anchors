"""Explicit classification and stage-aware discovery of canonical persistent state.

Canonical runs are found from manifests, output hashes and validation records on the
filesystem, never from run IDs written into code. Runtime dependencies are traced from the
actual consumers: Stage 4 labels read Stage 3 tables and a raw-resample run; Stage 5
rollout identity hashes the Stage 3 tables and the frozen target mapping (``pilot.inputs``);
Stage 6 activation runs hash the Stage 3 tables and the Stage 4 labels
(``features.extract.load_inputs``) and are the measured-fit admission evidence for Stage 7.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from preanchor.config import config_hash, read_yaml
from preanchor.ids import file_hash
from preanchor.io import UnitStore, read_json, validate_artifact, validate_parquet
from preanchor.logging import redact
from preanchor.manifest import load_manifest

MUST_BACK_UP = "MUST_BACK_UP"
SHOULD_BACK_UP = "SHOULD_BACK_UP"
REACQUIRE = "REACQUIRE"
CACHE_EXCLUDE = "CACHE_EXCLUDE"
HISTORICAL_EVIDENCE = "HISTORICAL_EVIDENCE"
SECRET = "SECRET"
CLASSIFICATIONS = (MUST_BACK_UP, SHOULD_BACK_UP, REACQUIRE, CACHE_EXCLUDE,
                   HISTORICAL_EVIDENCE, SECRET)

# Runtime input dependencies between canonical stages (consumer -> producers).
# Stage 6 reads Stage 3 and Stage 4; Stage 5 is a continuation dependency of the MVP after
# Stage 6 (the fresh pilot is a required experiment), so a Stage 6 bundle carries it too.
STAGE_DEPENDENCIES: dict[int, tuple[int, ...]] = {3: (), 4: (3,), 5: (3,), 6: (3, 4, 5)}
KNOWN_STAGES = tuple(sorted(STAGE_DEPENDENCIES))

STAGE3_FILES = ("trajectories.parquet", "sentences.parquet")
STAGE6_INPUTS = ("stage3/trajectories.parquet", "stage3/sentences.parquet",
                 "stage4/labels.parquet")
STAGE6_RECORDS = ("summary.json", "invariance_report.json", "resource_projection.json",
                  "boundary_audit.json")
ACTIVATION_UNITS = PurePosixPath("activations/units")
PILOT_TARGETS = "configs/pilot_targets_4682.yaml"
UNIT_NAMESPACES = {"units", "units_raw", "benchmark_units", "benchmark_units_raw"}
LIVE_STATES = {"prepared", "running", "stopped"}
GENERATED_STATES = {"generated"}
FINALIZED_STATES = {"labels_constructed", "adaptive_gate", "generated_and_scored"}

SECRET_NAME = re.compile(
    r"(^|/)(\.env(\..*)?|credentials[^/]*|secrets?[^/]*|[^/]*\.pem|[^/]*\.key|"
    r"id_(rsa|ed25519|ecdsa|dsa)[^/]*|\.netrc|\.git-credentials|\.ssh|\.aws)(/|$)", re.I)
SECRET_CONTENT = re.compile(r"\bhf_[A-Za-z0-9]{24,}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
                            r"\b(HF_TOKEN|WANDB_API_KEY|AWS_SECRET_ACCESS_KEY)=\S")
TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".log", ".md", ".patch", ".sha256", ".env"}
EXCLUDED_TOP_LEVEL = {"cache": CACHE_EXCLUDE, "wandb": CACHE_EXCLUDE, ".venv": CACHE_EXCLUDE,
                      "tmp": CACHE_EXCLUDE}


@dataclass
class Entry:
    path: str                      # logical path, POSIX, relative to PREANCHOR_ROOT
    classification: str
    stage: str                     # stage3 | stage4 | stage5 | stage2 | provenance | evidence
    required: bool                 # required for continuation at the selected stage
    validation: str                # sha256_sidecar | parquet | unit_file | run_manifest | bytes
    run_id: str | None = None
    unit: str | None = None        # logical path of the owning unit directory
    sha256: str | None = None
    size: int | None = None

    def record(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "size": self.size,
                "classification": self.classification, "stage": self.stage,
                "run_id": self.run_id, "required": self.required,
                "validation": self.validation, "unit": self.unit}


@dataclass
class Inventory:
    for_stage: int
    required_stages: tuple[int, ...]
    entries: list[Entry] = field(default_factory=list)
    reacquire: list[dict] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)          # run_id -> {"role", "stage", "manifest"}
    scientific_identity: dict = field(default_factory=dict)
    unit_counts: dict = field(default_factory=dict)

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def exclude(self, path: str, classification: str, reason: str, size: int | None = None) -> None:
        self.excluded.append({"path": path, "classification": classification,
                              "reason": reason, "bytes": size})


def stage_closure(for_stage: int) -> tuple[int, ...]:
    if for_stage not in STAGE_DEPENDENCIES:
        raise ValueError(f"Unknown stage {for_stage}; known: {KNOWN_STAGES}")
    seen: set[int] = set()
    pending = [for_stage]
    while pending:
        stage = pending.pop()
        if stage in seen:
            continue
        seen.add(stage)
        pending.extend(STAGE_DEPENDENCIES[stage])
    return tuple(sorted(seen))


def secret_reason(path: Path, logical: str) -> str | None:
    """Name- and content-based refusal; never ingest a credential."""
    if SECRET_NAME.search(logical):
        return "credential-like path"
    if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 4 * 1024 * 1024:
        text = path.read_bytes().decode("utf-8", errors="replace")
        if SECRET_CONTENT.search(text) or redact(text) != text:
            return "credential-like content"
    return None


def stage6_input_hashes(root: Path) -> dict[str, str] | None:
    """The exact keys ``features.extract.load_inputs`` records in every Stage 6 run."""
    processed = root / "data/processed"
    if not all((processed / name).is_file() for name in STAGE6_INPUTS):
        return None
    return {name: file_hash(processed / name) for name in STAGE6_INPUTS}


def stage3_hashes(root: Path, repo: Path) -> dict[str, str] | None:
    """The exact keys ``pilot.inputs`` hashes into every Stage 5 rollout ID."""
    base = root / "data/processed/stage3"
    if not all((base / name).is_file() for name in STAGE3_FILES):
        return None
    return {"stage3/trajectories.parquet": file_hash(base / "trajectories.parquet"),
            "stage3/sentences.parquet": file_hash(base / "sentences.parquet"),
            PILOT_TARGETS: file_hash(repo / PILOT_TARGETS)}


def _directory_bytes(path: Path) -> tuple[int, int]:
    """(files, bytes) via os.walk/scandir; symlinks are neither followed nor counted."""
    files = bytes_total = 0
    if not path.is_dir():
        return 0, 0
    for directory, _, names in os.walk(path):
        for name in names:
            item = Path(directory) / name
            if item.is_symlink():
                continue
            try:
                bytes_total += item.stat().st_size
            except OSError:
                continue
            files += 1
    return files, bytes_total


def _manifest_bytes(stage: Path) -> tuple[int, int] | None:
    """Payload size from a retained source manifest, avoiding a 45k-file NFS walk."""
    try:
        rows = json.loads((stage / "source_manifest.json").read_bytes())["files"]
        return len(rows), sum(int(r["size"]) for r in rows)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _run_files(run: Path):
    """Non-unit files of a run without descending into unit namespaces."""
    for directory, subdirs, names in os.walk(run):
        rel = Path(directory).relative_to(run)
        if len(rel.parts) == 2 and rel.parts[0] == "stages":
            subdirs[:] = [d for d in subdirs if d not in UNIT_NAMESPACES]
        if rel.parts == ("activations",):
            subdirs[:] = [d for d in subdirs if d != "units"]
        for name in names:
            yield Path(directory) / name


def load_runs(root: Path, inventory: Inventory) -> dict[str, dict]:
    runs = root / "artifacts/runs"
    loaded: dict[str, dict] = {}
    for directory in sorted(runs.iterdir()) if runs.is_dir() else []:
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            loaded[directory.name] = load_manifest(directory)
        except (ValueError, OSError, KeyError) as exc:
            inventory.warnings.append(f"run {directory.name}: unreadable manifest ({exc})")
            inventory.runs[directory.name] = {"role": "historical", "stage": "evidence",
                                              "reason": "manifest_invalid"}
    return loaded


def stage5_plan_path(run: Path) -> Path | None:
    for name in ("plan.json", "production_plan.json"):
        if (run / "stages/stage5" / name).is_file():
            return run / "stages/stage5" / name
    return None


def stage5_status(state: str | None) -> str:
    if state in FINALIZED_STATES:
        return "finalized"
    if state in GENERATED_STATES:
        return "generated"
    if state in LIVE_STATES:
        return "running" if state == "running" else "stopped"
    return "unknown"


def is_production(manifest: dict) -> bool:
    stage = manifest.get("stages", {}).get("stage5") or {}
    return stage.get("namespace") == "production" or stage.get("base_only") is True


def select_production(root: Path, manifests: dict[str, dict], hashes: dict[str, str] | None,
                      run_id: str | None, inventory: Inventory) -> str | None:
    candidates = []
    for name, manifest in manifests.items():
        if not is_production(manifest):
            continue
        if any((root / "artifacts/runs" / name).glob("SUPERSEDED*")):
            inventory.runs[name] = {"role": "historical", "stage": "evidence",
                                    "reason": "superseded_marker"}
            continue
        if hashes is not None and manifest.get("input_hashes") != hashes:
            inventory.runs[name] = {"role": "historical", "stage": "evidence",
                                    "reason": "input_hashes_differ_from_current_stage3"}
            continue
        candidates.append(name)
    if run_id is not None:
        if run_id not in candidates:
            inventory.blockers.append(f"--run-id {run_id} is not a current production run")
            return None
        return run_id
    if len(candidates) > 1:
        inventory.blockers.append(
            "Several production runs match the current Stage 3 inputs; pass --run-id: "
            + ", ".join(candidates))
        return None
    return candidates[0] if candidates else None


def _run_file(role: str, rel: PurePosixPath, finalized: bool) -> tuple[str, str, bool] | None:
    """(classification, validation, required) for a non-unit run file, or None to exclude."""
    parts = rel.parts
    name = rel.name
    sidecar = name.endswith(".sha256")
    if name == ".run.lock" or ".partial" in name:
        return None
    if role == "historical":
        return HISTORICAL_EVIDENCE, "bytes", False
    if parts[0] in ("manifest.json", "manifest.json.sha256"):
        return MUST_BACK_UP, "run_manifest", True
    if parts[0] == "manifests":
        return MUST_BACK_UP, "bytes" if sidecar else "sha256_sidecar", True
    if parts[0] in ("resolved_config.yaml", "resolved_config.yaml.sha256", "git.patch",
                    "git.patch.sha256"):
        return MUST_BACK_UP, "bytes" if sidecar else "sha256_sidecar", True
    if parts[0] in ("heartbeat.json", "failure.json") or name.startswith("SUPERSEDED"):
        return SHOULD_BACK_UP, "bytes", False
    if parts[0] in ("environment", "logs"):
        return SHOULD_BACK_UP, "bytes", False
    if parts[0] == "activations":
        if name.endswith(".parquet"):
            return MUST_BACK_UP, "parquet", True
        return MUST_BACK_UP, "bytes" if sidecar else "sha256_sidecar", True
    if parts[0] == "stages" and len(parts) >= 3:
        rest = parts[2:]
        if rest[0] in ("progress.json", "telemetry.json", "startup.log", "telemetry",
                       "failures"):
            return SHOULD_BACK_UP, "bytes", False
        if rest[0] == "attempts":
            return MUST_BACK_UP, "bytes" if sidecar else "sha256_sidecar", True
        if role in ("stage2",):
            return SHOULD_BACK_UP, "bytes" if sidecar else "sha256_sidecar", False
        if name.endswith(".parquet"):
            return MUST_BACK_UP, "parquet", True
        if sidecar or name.endswith(".schema.json"):
            return MUST_BACK_UP, "bytes" if sidecar else "sha256_sidecar", True
        return MUST_BACK_UP, "sha256_sidecar", True
    return SHOULD_BACK_UP, "bytes", False


def collect_run(root: Path, run_id: str, role: str, stage: str, inventory: Inventory, *,
                required_stage: bool, finalized: bool = True,
                expected_units: set[str] | None = None,
                measure_excluded: bool = False) -> dict:
    """Walk one run directory. Units are validated with ``UnitStore.validate`` and only
    complete, valid units become entries; partial or corrupt units are never ingested."""
    run = root / "artifacts/runs" / run_id
    counts = {"valid_units": 0, "partial_units": 0, "corrupt_units": 0, "reacquire_units": 0}
    for path in sorted(_run_files(run)):
        rel = PurePosixPath(path.relative_to(run).as_posix())
        logical = f"artifacts/runs/{run_id}/{rel}"
        if path.is_symlink():
            inventory.exclude(logical, CACHE_EXCLUDE, "symlink")
            continue
        if not path.is_file():
            continue
        secret = secret_reason(path, logical)
        if secret:
            inventory.blockers.append(f"{logical}: {secret}")
            inventory.exclude(logical, SECRET, secret, path.stat().st_size)
            continue
        decision = _run_file(role, rel, finalized)
        if decision is None:
            inventory.exclude(logical, CACHE_EXCLUDE, "operational lock or partial file",
                              path.stat().st_size)
            continue
        classification, validation, required = decision
        inventory.add(Entry(logical, classification, stage, required and required_stage,
                            validation, run_id=run_id, size=path.stat().st_size))
    namespaces = [d for d in sorted(run.glob("stages/*/*"))
                  if d.name in UNIT_NAMESPACES and d.is_dir()]
    if (run / ACTIVATION_UNITS).is_dir():
        namespaces.append(run / ACTIVATION_UNITS)
    for namespace_dir in namespaces:
        namespace_logical = f"artifacts/runs/{run_id}/{namespace_dir.relative_to(run).as_posix()}"
        if role in ("stage2", "stage4_raw", "historical"):
            measured = _manifest_bytes(namespace_dir.parent) if measure_excluded is False \
                else None
            files, size = measured or _directory_bytes(namespace_dir)
            counts["reacquire_units"] += files
            inventory.exclude(namespace_logical, REACQUIRE if role != "historical"
                              else HISTORICAL_EVIDENCE,
                              "bulk payload reacquired from the pinned upstream revision"
                              if role != "historical" else "historical run payload", size)
            continue
        store = UnitStore(namespace_dir)
        for unit in sorted(namespace_dir.iterdir()):
            unit_logical = f"{namespace_logical}/{unit.name}"
            if unit.name in (".locks", "quarantine") or not unit.is_dir():
                if unit.is_dir():
                    inventory.exclude(unit_logical, CACHE_EXCLUDE, "unit locks or quarantine")
                continue
            if not (unit / "SUCCESS.json").is_file():
                counts["partial_units"] += 1
                inventory.exclude(unit_logical, CACHE_EXCLUDE, "partial unit without SUCCESS")
                continue
            if expected_units is not None and unit.name not in expected_units:
                inventory.blockers.append(f"{unit_logical}: unit is not in the run plan")
                continue
            try:
                store.validate(unit.name)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                counts["corrupt_units"] += 1
                message = f"{unit_logical}: corrupt unit ({exc})"
                (inventory.blockers if finalized else inventory.warnings).append(message)
                inventory.exclude(unit_logical, CACHE_EXCLUDE, "corrupt unit rejected")
                continue
            counts["valid_units"] += 1
            for path in sorted(unit.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    logical = f"{unit_logical}/{path.relative_to(unit).as_posix()}"
                    inventory.add(Entry(logical, MUST_BACK_UP, stage, required_stage,
                                        "unit_file", run_id=run_id, unit=unit_logical,
                                        size=path.stat().st_size))
    inventory.unit_counts[run_id] = counts
    return counts


def collect_tree(root: Path, relative: str, classification: str, stage: str,
                 inventory: Inventory, *, required: bool = False) -> int:
    directory = root / relative
    count = 0
    for path in sorted(directory.rglob("*")) if directory.is_dir() else []:
        if path.is_symlink() or not path.is_file():
            continue
        logical = f"{relative}/{path.relative_to(directory).as_posix()}"
        secret = secret_reason(path, logical)
        if secret:
            inventory.blockers.append(f"{logical}: {secret}")
            inventory.exclude(logical, SECRET, secret, path.stat().st_size)
            continue
        name = path.name
        if name.endswith(".sha256"):
            validation = "bytes"
        elif name.endswith(".parquet") and Path(str(path) + ".schema.json").is_file():
            validation = "parquet"
        elif Path(str(path) + ".sha256").is_file():
            validation = "sha256_sidecar"
        else:
            validation = "bytes"
        inventory.add(Entry(logical, classification, stage, required, validation,
                            size=path.stat().st_size))
        count += 1
    return count


def validate_entry(root: Path, entry: Entry) -> str:
    """Artifact-specific validation, then the SHA-256 that names the content object."""
    path = root / entry.path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing: {entry.path}")
    if entry.validation == "sha256_sidecar":
        validate_artifact(path)
    elif entry.validation == "parquet":
        validate_parquet(path)
    elif entry.validation == "run_manifest":
        load_manifest(path.parent)
    elif entry.validation == "unit_file":
        pass   # the whole unit was validated by UnitStore.validate at collection
    elif entry.validation != "bytes":
        raise ValueError(f"unknown validation {entry.validation}")
    return file_hash(path)


def discover(root: Path, repo: Path, *, for_stage: int, run_id: str | None = None,
             include_historical: bool = False, measure_excluded: bool = False) -> Inventory:
    """``measure_excluded`` walks caches and reacquirable payloads for byte counts; it is
    off by default because a 45k-file NFS walk costs about a minute."""
    root, repo = Path(root), Path(repo)
    inventory = Inventory(for_stage=for_stage, required_stages=stage_closure(for_stage))
    required = set(inventory.required_stages)
    manifests = load_runs(root, inventory)

    def measured(path: Path) -> int | None:
        if not measure_excluded:
            return None
        return _directory_bytes(path)[1] if path.is_dir() else path.stat().st_size

    revisions: dict[str, set] = {}

    def note_revisions(manifest: dict) -> None:
        for key, value in (manifest.get("revisions") or {}).items():
            revisions.setdefault(key, set()).add(value)

    # ---- Stage 3 -------------------------------------------------------------------
    hashes = stage3_hashes(root, repo)
    if hashes is None:
        inventory.stages["stage3"] = {"status": "absent"}
        if 3 in required:
            inventory.blockers.append("Stage 3 canonical tables are absent "
                                      "(data/processed/stage3/*.parquet)")
    else:
        collect_tree(root, "data/processed/stage3", MUST_BACK_UP, "stage3", inventory,
                     required=3 in required)
        inventory.stages["stage3"] = {"status": "complete", "hashes": hashes}

    # ---- Stage 4 -------------------------------------------------------------------
    processed4 = root / "data/processed/stage4/labels.parquet"
    build_run = raw_run = None
    if processed4.is_file():
        labels_sha = file_hash(processed4)
        for name, manifest in manifests.items():
            stage = manifest.get("stages", {}).get("stage4_labels") or {}
            if (stage.get("state") == "complete"
                    and manifest.get("output_hashes", {}).get("stage4_labels") == labels_sha):
                build_run = name
        if build_run is None:
            inventory.warnings.append("No completed build run publishes the current "
                                      "data/processed/stage4/labels.parquet")
        else:
            raw_run = manifests[build_run].get("input_hashes", {}).get("raw_run_id")
            if raw_run not in manifests:
                inventory.warnings.append(f"Stage 4 raw run {raw_run} is not on this filesystem")
                raw_run = None
        collect_tree(root, "data/processed/stage4", MUST_BACK_UP, "stage4", inventory,
                     required=4 in required)
        stage4 = {"status": "complete", "labels_sha256": labels_sha, "build_run_id": build_run,
                  "raw_run_id": raw_run}
        try:
            from preanchor.labels.stage4 import labels_fingerprint

            stage4["fingerprint"] = labels_fingerprint(processed4)
        except (ValueError, OSError, KeyError, ImportError) as exc:
            inventory.blockers.append(f"Stage 4 labels do not validate: {exc}")
        inventory.stages["stage4"] = stage4
        if build_run:
            inventory.runs[build_run] = {"role": "stage4_build", "stage": "stage4"}
            note_revisions(manifests[build_run])
            collect_run(root, build_run, "stage4_build", "stage4", inventory,
                        required_stage=4 in required)
        if raw_run:
            inventory.runs[raw_run] = {"role": "stage4_raw", "stage": "stage4"}
            note_revisions(manifests[raw_run])
            collect_run(root, raw_run, "stage4_raw", "stage4", inventory,
                        required_stage=4 in required)
            raw_stage = root / "artifacts/runs" / raw_run / "stages/stage4_raw"
            units_files, units_bytes = _manifest_bytes(raw_stage) or (None, None)
            inventory.reacquire.append({
                "resource": f"artifacts/runs/{raw_run}/stages/stage4_raw/units",
                "kind": "stage4_raw_payload", "run_id": raw_run,
                "pin": {"dataset": manifests[raw_run]["revisions"]["dataset"],
                        "source_manifest_sha256": manifests[raw_run]["output_hashes"].get(
                            "stage4_raw_source_manifest"),
                        "raw_discovery": manifests[raw_run]["input_hashes"].get("raw_discovery")},
                "source": "configs/locks/raw_resamples_* via preanchor fetch-raw-resamples; "
                          "the retained source_manifest.json lists every upstream file hash",
                "files": units_files, "bytes": units_bytes,
                "reason": "bulk upstream payload; every file hash is recorded in the "
                          "retained source manifest and validated on reacquisition"})
    else:
        inventory.stages["stage4"] = {"status": "absent"}
        if 4 in required:
            inventory.blockers.append("Stage 4 canonical labels are absent "
                                      "(data/processed/stage4/labels.parquet)")

    # ---- Stage 2 (metadata only; payload reacquired) --------------------------------
    for name, manifest in manifests.items():
        stage = manifest.get("stages", {}).get("stage2") or {}
        if stage.get("state") == "complete" and not stage.get("smoke"):
            inventory.runs[name] = {"role": "stage2", "stage": "stage2"}
            note_revisions(manifest)
            collect_run(root, name, "stage2", "stage2", inventory, required_stage=False)
            units_files, units_bytes = _manifest_bytes(
                root / "artifacts/runs" / name / "stages/stage2") or (None, None)
            inventory.reacquire.append({
                "resource": f"artifacts/runs/{name}/stages/stage2/units",
                "kind": "stage2_payload", "run_id": name,
                "pin": {"dataset": manifest["revisions"]["dataset"],
                        "source_manifest_sha256": manifest["output_hashes"].get(
                            "stage2_source_manifest"),
                        "dataset_discovery": manifest["input_hashes"].get("dataset_discovery")},
                "source": "configs/locks/dataset_metadata.json via preanchor fetch",
                "files": units_files, "bytes": units_bytes,
                "reason": "pinned upstream dataset files; hashes retained in source_manifest"})

    # ---- Stage 5 -------------------------------------------------------------------
    production = select_production(root, manifests, hashes, run_id, inventory)
    stage5: dict = {"status": "absent"}
    if production:
        manifest = manifests[production]
        run = root / "artifacts/runs" / production
        state = manifest["stages"]["stage5"].get("state")
        status = stage5_status(state)
        plan_path = stage5_plan_path(run)
        expected: set[str] | None = None
        plan_sha = None
        if plan_path is None:
            inventory.blockers.append(f"{production}: Stage 5 plan is missing")
        else:
            try:
                validate_artifact(plan_path)
                plan = read_json(plan_path)
                expected = {j["rollout_id"] for j in plan["jobs"]}
                plan_sha = file_hash(plan_path)
            except (ValueError, OSError, KeyError) as exc:
                inventory.blockers.append(f"{production}: Stage 5 plan invalid ({exc})")
        finalized = status in ("generated", "finalized")
        inventory.runs[production] = {"role": "stage5_production", "stage": "stage5"}
        note_revisions(manifest)
        counts = collect_run(root, production, "stage5_production", "stage5", inventory,
                             required_stage=5 in required, finalized=finalized,
                             expected_units=expected)
        raw_root = run / "stages/stage5/units_raw"
        raw_valid = {p.parent.name for p in raw_root.glob("*/SUCCESS.json")} if expected else set()
        protected = len(raw_valid & expected) if expected else 0
        if finalized and expected and raw_valid != expected:
            inventory.blockers.append(
                f"{production}: {len(expected - raw_valid)} planned units lack raw completions "
                "although the run reports generation complete")
        admission = [name for name in ("runtime_admission.json", "tokenizer_validation.json")
                     if (run / "stages/stage5" / name).is_file()]
        resumable = plan_path is not None and len(admission) == 2
        rollout_hash = config_hash(sorted(expected)) if expected else None
        benchmarks: list[str] = []
        for name in ("resource_projection_a6000.json", "resource_projection.json"):
            projection_path = run / "stages/stage5" / name
            if projection_path.is_file():
                try:
                    projection = json.loads(projection_path.read_bytes())
                except ValueError:
                    continue
                for candidate in [projection.get("selected_benchmark_run"),
                                  projection.get("empirical_run")] + [
                        Path(e["path"]).parents[2].name for e in projection.get("evidence", [])
                        if isinstance(e, dict) and "path" in e]:
                    if candidate and candidate in manifests and candidate not in benchmarks:
                        benchmarks.append(candidate)
        for benchmark in benchmarks:
            inventory.runs[benchmark] = {"role": "stage5_benchmark", "stage": "stage5"}
            note_revisions(manifests[benchmark])
            collect_run(root, benchmark, "stage5_benchmark", "stage5", inventory,
                        required_stage=False, finalized=True)
        frozen = {}
        try:
            resolved = read_yaml(run / "resolved_config.yaml")
            from preanchor.labels.a6000 import MAX_MODEL_LEN
            from preanchor.labels.generate import ADD_SPECIAL_TOKENS

            frozen = {"model_revision": resolved["model"]["revision"],
                      "dtype": resolved["model"]["dtype"],
                      "seed_root": resolved["seed_root"],
                      "temperature": resolved["generation"]["temperature"],
                      "top_p": resolved["generation"]["top_p"],
                      "max_new_tokens": resolved["generation"]["max_new_tokens"],
                      "add_special_tokens": ADD_SPECIAL_TOKENS,
                      "max_model_len": int(MAX_MODEL_LEN),
                      "chat_template": resolved["model"]["chat_template"],
                      "quantization": resolved["model"]["quantization"]}
        except (ValueError, OSError, KeyError, ImportError) as exc:
            inventory.blockers.append(f"{production}: resolved config unreadable ({exc})")
        stage5 = {"status": status, "manifest_state": state, "run_id": production,
                  "planned_units": len(expected) if expected else None,
                  "protected_completed_units": protected, "resumable": resumable,
                  "finalized": status == "finalized", "plan_sha256": plan_sha,
                  "rollout_id_set_sha256": rollout_hash, "input_hashes": manifest["input_hashes"],
                  "benchmark_run_ids": benchmarks, "frozen": frozen,
                  "unit_counts": counts}
        if hashes is not None and manifest.get("input_hashes") != hashes:
            inventory.blockers.append(f"{production}: input hashes differ from Stage 3")
    elif 5 in required:
        unreadable = sorted(name for name, info in inventory.runs.items()
                            if info.get("reason") == "manifest_invalid")
        inventory.blockers.append(
            "No current Stage 5 production run matches the Stage 3 inputs"
            + (f"; runs with an unreadable or inconsistent manifest: {unreadable}"
               if unreadable else ""))
    inventory.stages["stage5"] = stage5

    # ---- Stage 6 (activation-extraction smoke: Stage 7 admission evidence) ---------
    stage6 = discover_stage6(root, manifests, inventory, required=6 in required,
                             note_revisions=note_revisions)
    inventory.stages["stage6"] = stage6

    # ---- Provenance and evidence trees ---------------------------------------------
    collect_tree(root, "artifacts/reconstruction", MUST_BACK_UP, "provenance", inventory)
    collect_tree(root, "artifacts/amendments", MUST_BACK_UP, "provenance", inventory)
    collect_tree(root, "artifacts/h100-admission", SHOULD_BACK_UP, "evidence", inventory)
    collect_tree(root, "artifacts/doctor", SHOULD_BACK_UP, "evidence", inventory)
    digest_note = root / "artifacts/vllm-image-digest.txt"
    if digest_note.is_file():
        inventory.add(Entry("artifacts/vllm-image-digest.txt", SHOULD_BACK_UP, "evidence",
                            False, "bytes", size=digest_note.stat().st_size))
    backups = root / "artifacts/backups"
    for live in sorted(backups.glob("stage5-live-*")) if backups.is_dir() else []:
        collect_tree(root, f"artifacts/backups/{live.name}/checkpoints", SHOULD_BACK_UP,
                     "evidence", inventory)
        pointer = live / "wandb/latest.json"
        if pointer.is_file():
            inventory.add(Entry(f"artifacts/backups/{live.name}/wandb/latest.json",
                                SHOULD_BACK_UP, "evidence", False, "bytes",
                                size=pointer.stat().st_size))
        for duplicate in ("run", "wandb"):
            if (live / duplicate).is_dir():
                inventory.exclude(f"artifacts/backups/{live.name}/{duplicate}", CACHE_EXCLUDE,
                                  "primary-only checkpoint copy; payload protected via the run",
                                  measured(live / duplicate))
    for legacy in sorted(backups.iterdir()) if backups.is_dir() else []:
        if legacy.is_dir() and not legacy.name.startswith("stage5-live-"):
            inventory.exclude(f"artifacts/backups/{legacy.name}", HISTORICAL_EVIDENCE,
                              "legacy snapshot; ingest with `backup import-legacy`",
                              measured(legacy))

    # ---- Historical runs -----------------------------------------------------------
    for name in manifests:
        if name not in inventory.runs:
            inventory.runs[name] = {"role": "historical", "stage": "evidence",
                                    "reason": "not a dependency of canonical continuation"}
    for name, info in inventory.runs.items():
        if info["role"] != "historical":
            continue
        if include_historical and name in manifests:
            collect_run(root, name, "historical", "evidence", inventory, required_stage=False)
        else:
            inventory.exclude(f"artifacts/runs/{name}", HISTORICAL_EVIDENCE,
                              info.get("reason", "historical"),
                              measured(root / "artifacts/runs" / name))

    # ---- Caches, self, secrets at the root ----------------------------------------
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.name in EXCLUDED_TOP_LEVEL:
            inventory.exclude(child.name, EXCLUDED_TOP_LEVEL[child.name],
                              "reacquirable cache or transient state", measured(child))
        elif child.is_file() and secret_reason(child, child.name):
            inventory.exclude(child.name, SECRET, "credential-like file at root")
    if (root / "artifacts/portable").is_dir():
        inventory.exclude("artifacts/portable", CACHE_EXCLUDE, "the portable store itself")
    cache_hub = root / "cache/huggingface"
    if cache_hub.is_dir():
        for item in sorted(cache_hub.iterdir()):
            inventory.exclude(f"cache/huggingface/{item.name}", CACHE_EXCLUDE,
                              "Hugging Face cache", measured(item))

    # ---- Pinned external resources ------------------------------------------------
    pins = {key: sorted(values) for key, values in revisions.items()}
    for key, values in pins.items():
        if len(values) > 1:
            inventory.blockers.append(f"Canonical runs disagree on the {key} revision: {values}")
    single = {key: values[0] for key, values in pins.items() if len(values) == 1}
    try:
        from preanchor.config import load_config

        config = load_config(repo / "configs/mvp.yaml", repo_root=repo)
        model_id, dataset_id = config.model.id, config.dataset.id
        minilm_id = config.labels["similarity_model"]
        vllm_image = config.generation["vllm_image"]
    except (ValueError, OSError, KeyError) as exc:
        inventory.warnings.append(f"configs/mvp.yaml unreadable: {exc}")
        model_id = dataset_id = minilm_id = vllm_image = None
    from preanchor.labels.pilot import DIGEST

    hub_dir = "cache/huggingface/hub"
    inventory.reacquire.extend([
        {"resource": f"{hub_dir}/models--{(model_id or '').replace('/', '--')}",
         "kind": "model_weights", "pin": {"id": model_id, "revision": single.get("model")},
         "source": "configs/mvp.yaml model.revision; run manifest revisions.model",
         "reason": "immutable Hugging Face snapshot (~28 GB)"},
        {"resource": f"{hub_dir}/models--{(model_id or '').replace('/', '--')}",
         "kind": "tokenizer", "pin": {"id": model_id, "revision": single.get("tokenizer")},
         "source": "configs/mvp.yaml model.tokenizer_revision",
         "reason": "part of the pinned model snapshot"},
        {"resource": f"{hub_dir}/models--{(minilm_id or '').replace('/', '--')}",
         "kind": "minilm_weights", "pin": {"id": minilm_id, "revision": single.get("embedding")},
         "source": "configs/mvp.yaml labels.similarity_revision",
         "reason": "immutable Hugging Face snapshot (~88 MB)"},
        {"resource": "docker image preanchor vLLM", "kind": "vllm_image",
         "pin": {"image": vllm_image, "digest": DIGEST},
         "source": "preanchor.labels.pilot.DIGEST; configs/mvp.yaml generation.vllm_image",
         "reason": "pulled by digest with scripts/start_vllm.sh"},
        {"resource": "cache/huggingface/datasets-hub", "kind": "dataset_transit_cache",
         "pin": {"id": dataset_id, "revision": single.get("dataset")},
         "source": "configs/mvp.yaml dataset.revision",
         "reason": "transit cache; canonical copies live in run units"},
    ])

    # ---- Scientific identity --------------------------------------------------------
    identity = {"stage3": (hashes or {}), "stage4": {k: v for k, v in inventory.stages.get(
        "stage4", {}).items() if k != "status"},
        "stage5": {k: stage5.get(k) for k in ("run_id", "input_hashes", "plan_sha256",
                                              "rollout_id_set_sha256", "frozen",
                                              "benchmark_run_ids")} if production else {},
        "stage6": {k: v for k, v in stage6.items() if k != "status"}
        if stage6.get("status") == "complete" else {},
        "revisions": single, "vllm_digest": DIGEST,
        "resolved_config_hashes": {}, "scientific_config_hashes": {},
        "producing_commits": {}, "dirty_diff_hashes": {}, "lock_hashes": {}, "run_ids": []}
    for name, info in sorted(inventory.runs.items()):
        if info["role"] == "historical" or name not in manifests:
            continue
        manifest = manifests[name]
        identity["run_ids"].append(name)
        identity["resolved_config_hashes"][name] = manifest["identity"]["resolved_config_hash"]
        identity["scientific_config_hashes"][name] = manifest["identity"][
            "scientific_config_hash"]
        identity["producing_commits"][name] = manifest["source"]["git_sha"]
        identity["dirty_diff_hashes"][name] = manifest["source"]["dirty_diff_hash"]
        identity["lock_hashes"][name] = manifest["source"]["lock_hash"]
    try:
        upstream = read_yaml(repo / "configs/locks/upstream.yaml")
        identity["upstream"] = {name: entry["revision"]
                                for name, entry in upstream["repositories"].items()}
    except (ValueError, OSError, KeyError):
        identity["upstream"] = {}
    inventory.scientific_identity = identity
    return inventory


def stage6_run_summary(run: Path) -> dict:
    """Validate the records a completed Stage 6 run must carry and extract its identity."""
    stage = run / "stages/stage6"
    for name in STAGE6_RECORDS:
        validate_artifact(stage / name)
    summary = read_json(stage / "summary.json")
    report = read_json(stage / "invariance_report.json")
    projection = read_json(stage / "resource_projection.json")
    index = validate_parquet(run / "activations/index.parquet")
    units = {u["unit_key"]: u for u in summary["units"]}
    if index.num_rows != summary["n_rows"] or len(units) != summary["n_units"]:
        raise ValueError("index rows or unit count differ from summary.json")
    keys = {p.split("/")[1] for p in index.column("shard_path").to_pylist()}
    if keys != set(units):
        raise ValueError("index shard paths do not match the summary's units")
    if not report.get("primary_pass") or not report.get("logit_identity_pass"):
        raise ValueError("invariance report does not pass the primary gate")
    return {"n_units": summary["n_units"], "n_rows": summary["n_rows"],
            "trajectories": sorted(u["trajectory_id"] for u in units.values()),
            "index_sha256": summary["index_sha256"],
            "shard_sha256": {u["trajectory_id"]: u["shard_sha256"] for u in units.values()},
            "unit_keys": sorted(units), "activation_site": summary["activation_site"],
            "n_layers": summary["architecture"]["n_layers"],
            "d_model": summary["architecture"]["d_model"],
            "invariance": {k: report.get(k) for k in ("primary_pass", "logit_identity_pass",
                                                      "n_boundaries", "diagnostic_max_abs",
                                                      "diagnostic_within_atol", "atol")},
            "projection": {k: projection.get(k) for k in ("stage", "unit_count", "row_count",
                                                          "token_count", "projected_seconds",
                                                          "projected_bytes", "code_commit")}}


def discover_stage6(root: Path, manifests: dict[str, dict], inventory: Inventory, *,
                    required: bool, note_revisions) -> dict:
    """Every complete Stage 6 run whose recorded inputs are the current Stage 3/4 tables is
    canonical admission evidence; incomplete or stale runs stay historical."""
    expected = stage6_input_hashes(root)
    runs: dict[str, dict] = {}
    for name, manifest in sorted(manifests.items()):
        stage = manifest.get("stages", {}).get("stage6")
        if not stage:
            continue
        if stage.get("state") != "complete":
            inventory.runs[name] = {"role": "historical", "stage": "evidence",
                                    "reason": f"stage6_state_{stage.get('state')}"}
            continue
        if expected is None or manifest.get("input_hashes") != expected:
            inventory.runs[name] = {"role": "historical", "stage": "evidence",
                                    "reason": "input_hashes_differ_from_current_stage3_4"}
            continue
        run = root / "artifacts/runs" / name
        try:
            summary = stage6_run_summary(run)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            inventory.blockers.append(f"{name}: Stage 6 records invalid ({exc})")
            continue
        if sorted(stage.get("trajectories") or []) != summary["trajectories"] \
                or summary["n_rows"] != stage.get("n_rows"):
            inventory.blockers.append(f"{name}: Stage 6 outputs differ from the prepared scope")
            continue
        inventory.runs[name] = {"role": "stage6_smoke", "stage": "stage6"}
        note_revisions(manifest)
        counts = collect_run(root, name, "stage6_smoke", "stage6", inventory,
                             required_stage=required, finalized=True,
                             expected_units=set(summary["unit_keys"]))
        if counts["valid_units"] != summary["n_units"]:
            inventory.blockers.append(f"{name}: {summary['n_units'] - counts['valid_units']} "
                                      "activation units are missing or invalid")
        runs[name] = summary
    if not runs:
        if required:
            inventory.blockers.append("No complete Stage 6 run matches the current Stage 3/4 "
                                      "tables (activation smoke admission evidence)")
        return {"status": "absent"}
    return {"status": "complete", "run_ids": sorted(runs), "input_hashes": expected,
            "runs": runs, "total_rows": sum(r["n_rows"] for r in runs.values()),
            "total_units": sum(r["n_units"] for r in runs.values())}


def highest_protected_stage(inventory: Inventory) -> int | None:
    protected = None
    for stage in KNOWN_STAGES:
        if stage > inventory.for_stage:
            break
        status = inventory.stages.get(f"stage{stage}", {}).get("status")
        if status in ("complete", "generated", "finalized"):
            protected = stage
    return protected


def bundle_kind(inventory: Inventory) -> str:
    status = inventory.stages.get(f"stage{inventory.for_stage}", {}).get("status")
    return "live_checkpoint" if status in ("running", "stopped") else "canonical"
