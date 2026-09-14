from pathlib import Path

import pytest
import yaml

from preanchor.config import ExperimentConfig, Settings, Splits, load_config, read_yaml

ROOT = Path(__file__).resolve().parents[2]


def test_mvp_scientific_pins_and_resolved_split():
    config = load_config(ROOT / "configs/mvp.yaml")
    assert config.model.revision == "1df8507178afcc1bef68cd8c393f61a886323761"
    assert config.dataset.revision == "a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2"
    assert config.model.quantization is None
    assert config.features["layers"] == list(range(49))
    assert config.splits.test == [6998, 2236, 2870, 3916]
    assert config.labels["smoothing_alpha"] == 1e-9
    assert config.generation["samples_per_fork"] == 32
    assert config.generation["adaptive_samples_per_fork"] == 100


def test_smoke_preserves_science_and_limits_work():
    main = load_config(ROOT / "configs/mvp.yaml").model_dump()
    smoke = load_config(ROOT / "configs/smoke.yaml").model_dump()
    assert smoke["execution"]["trajectory"] == "330:correct"
    assert smoke["execution"]["limit_sentences"] == 16
    del main["execution"], smoke["execution"]
    assert main == smoke


def test_hash_uses_resolved_content_and_excludes_display_only_from_science():
    original = load_config(ROOT / "configs/mvp.yaml")
    data = original.model_dump()
    data["tracking"]["streamlit_port"] = 8502
    display = ExperimentConfig.model_validate(data)
    assert original.resolved_hash() != display.resolved_hash()
    assert original.scientific_hash() == display.scientific_hash()
    data["labels"]["smoothing_alpha"] = 0.5
    scientific = ExperimentConfig.model_validate(data)
    assert original.scientific_hash() != scientific.scientific_hash()


@pytest.mark.parametrize("groups", [
    {"train": [1, 1], "validation": [2], "test": [3]},
    {"train": [1], "validation": [1], "test": [3]},
    {"train": [], "validation": [2], "test": [3]},
])
def test_split_rejects_leakage(groups):
    with pytest.raises(ValueError):
        Splits(schema_version=1, **groups)


def test_rejects_unknown_top_level_and_invalid_invariants():
    config = load_config(ROOT / "configs/mvp.yaml").model_dump()
    for section, key, value in [("model", "quantization", "int8"),
                                ("features", "layers", [48]),
                                ("evaluation", "test_locked", False),
                                ("probes", "group_column", "sentence_id")]:
        changed = load_config(ROOT / "configs/mvp.yaml").model_dump()
        changed[section][key] = value
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate(changed)
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({**config, "typo": True})


def test_yaml_rejects_duplicate_keys_and_cycles(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("labels: 1\nlabels: 2\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        read_yaml(path)
    path.write_text("extends: config.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(path)


def test_split_contents_are_in_hash(tmp_path):
    (tmp_path / "configs/splits").mkdir(parents=True)
    split = read_yaml(ROOT / "configs/splits/mvp_20_v1.yaml")
    split["train"][0], split["test"][0] = split["test"][0], split["train"][0]
    (tmp_path / "configs/splits/mvp_20_v1.yaml").write_text(yaml.safe_dump(split))
    original = load_config(ROOT / "configs/mvp.yaml")
    changed = load_config(ROOT / "configs/mvp.yaml", repo_root=tmp_path)
    assert original.scientific_hash() != changed.scientific_hash()


def test_persistent_settings_reject_ephemeral_and_escaping_symlinks(tmp_path):
    with pytest.raises(ValueError, match="explicitly"):
        Settings(root=None, artifacts=None).persistent_root()
    link = tmp_path / "persistent"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="beneath"):
        Settings(root=link, artifacts=None).persistent_root()
    with pytest.raises(ValueError, match="PREANCHOR_ARTIFACTS"):
        Settings(root=Path("/lambda/nfs/preanchor/preanchor"),
                 artifacts=tmp_path).persistent_root()
