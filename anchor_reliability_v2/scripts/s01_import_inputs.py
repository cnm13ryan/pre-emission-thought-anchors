"""Copy read-only MVP inputs into the study data root with full provenance.

Copied: Stage 3 trajectories/sentences, Stage 4 labels (as the MVP reference to
reproduce), Stage 7 boundary features, the Stage 4 raw-resample source manifest (the
upstream file list and hashes used to reacquire rollouts), and the fixed split.
Referenced in place (checksummed, never written): the Stage 7 activation shards, 3.1 GB.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2.paths import INPUTS, MVP_ROOT, STUDY, sha256_file, write_json  # noqa: E402

MVP_REPO = STUDY.parent
COPY = {
    "trajectories.parquet": MVP_ROOT / "data/processed/stage3/trajectories.parquet",
    "sentences.parquet": MVP_ROOT / "data/processed/stage3/sentences.parquet",
    "mvp_labels.parquet": MVP_ROOT / "data/processed/stage4/labels.parquet",
    "mvp_features.parquet": MVP_ROOT / "data/processed/stage7/features.parquet",
    "raw_source_manifest.json": MVP_ROOT / "artifacts/runs/20260915T134433Z-d7d055d8b7-05a81618"
    / "stages/stage4_raw/source_manifest.json",
    "mvp_split.yaml": MVP_REPO / "configs/splits/mvp_20_v1.yaml",
}
ACTIVATIONS = MVP_ROOT / "artifacts/runs/20260915T182306Z-3ab408d80a-7452f1c2/activations"


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name, src in COPY.items():
        dst = INPUTS / name
        shutil.copy2(src, dst)
        digest = sha256_file(dst)
        sidecar = Path(str(src) + ".sha256")
        recorded = sidecar.read_text().split()[0] if sidecar.exists() else None
        if recorded and recorded != digest:
            raise SystemExit(f"{name}: copy does not match the MVP sidecar hash")
        copied[name] = {"source": str(src), "sha256": digest,
                        "mvp_sidecar_sha256": recorded, "bytes": dst.stat().st_size}
    shards = {str(p.relative_to(ACTIVATIONS)): sha256_file(p)
              for p in sorted(ACTIVATIONS.glob("units/*/*.safetensors"))}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=MVP_REPO, capture_output=True,
                            text=True).stdout.strip()
    write_json(INPUTS / "MANIFEST.json", {
        "schema_version": 1, "purpose": "anchor_reliability_v2 read-only inputs",
        "mvp_reference_commit": "f11823f", "branch_head_at_import": commit,
        "copied": copied,
        "referenced_in_place": {"activation_root": str(ACTIVATIONS), "shards": shards,
                                "index_sha256": sha256_file(ACTIVATIONS / "index.parquet")},
        "upstream_dataset": {"id": "uzaymacar/math-rollouts",
                             "revision": "a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2"},
        "model": {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                  "revision": "1df8507178afcc1bef68cd8c393f61a886323761"}})
    print({k: v["sha256"][:12] for k, v in copied.items()}, len(shards), "shards")


if __name__ == "__main__":
    main()
