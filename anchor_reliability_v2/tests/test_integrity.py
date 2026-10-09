"""Data integrity and reproducibility. Data-dependent tests skip when $ARV2_DATA is absent."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arv2 import paths  # noqa: E402
from arv2.vllm_client import sample_seed  # noqa: E402

HAS_DATA = (paths.INPUTS / "MANIFEST.json").exists()


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed_derivations_are_stable():
    # values pinned at plan v1; any change breaks replicate identity
    assert sample_seed(20260914, "x", 1, 2) == sample_seed(20260914, "x", 1, 2)
    s04 = _load("s04_scores")
    m1, m2 = s04.half_mask("330:correct", 3, 100), s04.half_mask("330:correct", 3, 100)
    assert (m1 == m2).all() and m1.sum() == 50
    assert not (s04.half_mask("330:correct", 4, 100) == m1).all()


@pytest.mark.skipif(not HAS_DATA, reason="study data root not present")
def test_inputs_match_manifest_hashes():
    manifest = paths.verify_inputs()
    assert manifest["copied"]["mvp_labels.parquet"]["sha256"].startswith("fab9886e")
    assert len(manifest["referenced_in_place"]["shards"]) == 40


@pytest.mark.skipif(not (ROOT / "results/s04_summary.json").exists(),
                    reason="s04 not run")
def test_mvp_labels_reproduced_exactly():
    s = json.loads((ROOT / "results/s04_summary.json").read_text())
    assert s["reproduction"]["passed"] and s["reproduction"]["rows"] == 4019
    assert s["rollouts"] == 647400


@pytest.mark.skipif(not (paths.DERIVED / "sentence_scores.parquet").exists(),
                    reason="s04 not run")
def test_scores_recompute_deterministically():
    import pandas as pd

    from arv2 import scores as S
    d = pd.read_parquet(paths.DERIVED / "sentence_scores.parquet")
    r = pd.read_parquet(paths.DERIVED / "rollouts_v2.parquet")
    row = d[d.label_status == "OK"].iloc[123]
    F = r[(r.trajectory_id == row.trajectory_id) & (r.fork_idx == row.sentence_idx)]
    G = r[(r.trajectory_id == row.trajectory_id) & (r.fork_idx == row.sentence_idx + 1)]
    w = F[F.dissimilar]
    assert np.isclose(S.total_variation(w.cat.tolist(), G.cat.tolist()), row.tv)
    assert np.isclose(S.total_variation(F.cat.tolist(), G.cat.tolist()), row.fork_tv)
