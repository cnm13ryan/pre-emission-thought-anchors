"""Study locations. Nothing here points into the MVP code; MVP *data* is read-only input."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ARV2_DATA", Path.home() / "preanchor-root/anchor_reliability_v2"))
MVP_ROOT = Path(os.environ.get("ARV2_MVP_ROOT", Path.home() / "preanchor-root"))
INPUTS = DATA / "inputs"
DERIVED = DATA / "derived"
RESULTS = STUDY / "results"

SEED_ROOT = 20260914
SPLIT = {"train": [2238, 1591, 4682, 6596, 3448, 2189, 3550, 4605, 2137, 4019, 3360, 3935],
         "validation": [330, 4164, 2050, 6481], "test": [6998, 2236, 2870, 3916]}
PROBLEMS = sorted(p for ps in SPLIT.values() for p in ps)
MVP_THRESHOLD = 1.3549810893213545


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def verify_inputs() -> dict:
    """Fail closed if any copied input differs from its recorded hash."""
    manifest = json.loads((INPUTS / "MANIFEST.json").read_text())
    for name, rec in manifest["copied"].items():
        if sha256_file(INPUTS / name) != rec["sha256"]:
            raise ValueError(f"input changed: {name}")
    return manifest
