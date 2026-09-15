import json
from pathlib import Path
from unittest.mock import Mock

from typer.testing import CliRunner

from preanchor.cli import app
from preanchor.config import Settings, load_config, read_yaml
from preanchor.doctor import check_submodules, run_doctor

ROOT = Path(__file__).resolve().parents[2]


def test_cpu_doctor_never_touches_gpu_network_or_persistence(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("production checks must not run"))
    for name in ["check_gpu", "check_persistence", "check_model", "check_tracking"]:
        monkeypatch.setattr(f"preanchor.doctor.{name}", forbidden)
    result = CliRunner().invoke(app, ["doctor", "--cpu-only", "--config",
                                     str(ROOT / "configs/mvp.yaml")])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["ok"] is True
    assert report["gpu_stage_admitted"] is False
    assert report["scope"] == "cpu_bootstrap"
    assert len(report["unchecked"]) == 4
    forbidden.assert_not_called()


def test_production_doctor_cannot_admit_failed_check(monkeypatch):
    monkeypatch.setattr("preanchor.doctor.check_gpu", Mock(side_effect=RuntimeError("secret")))
    for name in ["check_persistence", "check_model", "check_tracking"]:
        monkeypatch.setattr(f"preanchor.doctor.{name}", lambda *args: {"fixture": True})
    report = run_doctor(load_config(ROOT / "configs/mvp.yaml"), Settings(), ROOT)
    assert report["ok"] is False
    assert report["gpu_stage_admitted"] is False
    assert "secret" not in json.dumps(report)


def test_missing_submodule_rejected_even_inside_parent_repo(tmp_path):
    (tmp_path / "configs/locks").mkdir(parents=True)
    (tmp_path / "configs/locks/upstream.yaml").write_text(
        "repositories:\n  vendor/missing:\n    revision: abc\n")
    assert check_submodules(tmp_path)[0]["ok"] is False


def test_hardware_profiles_match_spec():
    for name, seqs, utilization, length in [
        ("h100_pcie_80", 4, 0.90, 32768),
        ("a6000_48", 2, 0.92, 20480),
        ("a100_40", 1, 0.92, 20480),
    ]:
        value = read_yaml(ROOT / f"configs/lambda/{name}.yaml")
        assert value["vllm"] == {"max_num_seqs": seqs,
                                  "gpu_memory_utilization": utilization,
                                  "max_model_len": length}
        assert value["extraction_batch_size"] == 1
