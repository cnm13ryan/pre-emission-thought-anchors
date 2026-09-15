"""A complete synthetic Stage 7 -> Stage 8 lineage for Stage 9 tests (no network, tiny
model). Train 5 problems, validation 4, test 4; eight sentences per trajectory with three
anchors each, so validation holds 24 positives and the Platt calibrator is fitted."""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
from tiny_qwen2 import tiny_loader, tiny_model, tiny_tokenizer, write_synthetic_root

from preanchor.features.dataset import build_dataset
from preanchor.features.extract import execute as extract_execute
from preanchor.features.extract import load_inputs, prepare, select_work
from preanchor.features.observables import head_from_model
from preanchor.io import validate_parquet, write_parquet
from preanchor.probes import fit, select
from preanchor.probes.validate import validate_probes

SOURCE = {"git_sha": "a" * 40, "dirty_diff_hash": "b" * 64, "lock_hash": "c" * 64,
          "package_freeze": []}
TRAIN = [2238, 1591, 4682, 6596, 3448]
VALIDATION = [330, 4164, 2050, 6481]
TEST = [6998, 2236, 2870, 3916]
LIMIT = 8
ANCHORS = {1, 3, 5}


def relabel(root: Path) -> None:
    path = root / "data/processed/stage4/labels.parquet"
    table = validate_parquet(path)
    rows = table.to_pylist()
    for r in rows:
        r["anchor"] = r["sentence_idx"] in ANCHORS
        r["importance_primary"] = 0.05 * r["sentence_idx"] + (1.0 if r["anchor"] else 0.0)
        r["importance_log1p"] = math.log1p(r["importance_primary"])
    for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
        Path(str(path) + suffix).unlink()
    write_parquet(path, pa.Table.from_pylist(rows, schema=table.schema))


def build_lineage(base: Path, repo: Path, config, *, problems=None) -> tuple[Path, Path, Path]:
    """Returns ``(root, stage7_run, stage8_run)``; Stage 8 is selected and validated."""
    root = base / "root"
    write_synthetic_root(root, config)
    relabel(root)
    work = [f"{p}:{k}" for p in (problems or TRAIN + VALIDATION + TEST)
            for k in ("correct", "incorrect")]
    inputs, hashes = load_inputs(root)
    s7 = prepare(root, repo, config, "test", select_work(config, inputs, work, LIMIT), hashes)
    extract_execute(s7, repo, config, trajectories=work, limit_sentences=LIMIT,
                    tracking=False, loader=tiny_loader, diagnostics=False)
    build_dataset(s7, repo, config, root, device="cpu", tracking=False,
                  head_loader=lambda c, d: head_from_model(tiny_model(), d),
                  tokenizer_loader=lambda c: tiny_tokenizer())
    s8 = fit.prepare_run(root, repo, config, s7, "fit")
    fit.execute(s8, repo, config, tracking=False, workers=2)
    select.execute(s8, repo, config, workers=2)
    report = validate_probes(s8, config, root)
    assert report["ok"] and report["selected"]
    return root, s7, s8
