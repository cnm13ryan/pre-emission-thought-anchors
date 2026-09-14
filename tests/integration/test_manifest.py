from pathlib import Path
from types import SimpleNamespace

import pytest

from preanchor.config import load_config
from preanchor.manifest import check_resume, create_run, load_manifest
from preanchor.smoke import run_smoke
from preanchor.tracking import start_tracking

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = {'git_sha': 'a' * 40, 'dirty_diff_hash': 'b' * 64,
              'lock_hash': 'c' * 64, 'package_freeze': []}
    monkeypatch.setattr('preanchor.manifest.provenance', lambda repo: (source.copy(), b''))
    config = load_config(ROOT / 'configs/smoke.yaml')
    directory = create_run(tmp_path, ROOT, config, ['fixture'])
    return directory, config, source


def test_run_provenance_resume_and_all_formats(prepared):
    directory, config, _ = prepared
    first = run_smoke(directory, ROOT, config, tracking=False)
    assert first['created'] == 3
    manifest = load_manifest(directory)
    assert manifest['stages']['stage1']['state'] == 'complete'
    assert manifest['gpu_gate'].startswith('paused')
    assert manifest['revisions']['tokenizer'] == config.model.tokenizer_revision
    second = run_smoke(directory, ROOT, config, resume=True, tracking=False)
    assert second['skipped'] == 3
    with pytest.raises(ValueError, match='same unit count'):
        run_smoke(directory, ROOT, config, resume=True, units=4, tracking=False)


def test_resume_refuses_code_config_lock_changes(prepared):
    directory, config, source = prepared
    original = config.model_copy(deep=True)
    config.labels['smoothing_alpha'] = 0.5
    with pytest.raises(ValueError, match='Resume refused'):
        check_resume(directory, ROOT, config)
    for key in ('git_sha', 'lock_hash', 'dirty_diff_hash'):
        old = source[key]
        source[key] = 'changed'
        with pytest.raises(ValueError, match='Resume refused'):
            check_resume(directory, ROOT, original)
        source[key] = old


def test_manifest_pointer_publication_recovers_from_stale_sidecar(prepared):
    directory, _, _ = prepared
    (directory / 'manifest.json.sha256').write_text('old-sidecar')
    assert load_manifest(directory)['schema_version'] == 1
    (directory / 'manifest.json').write_text('{}')
    with pytest.raises((ValueError, FileNotFoundError)):
        load_manifest(directory)


def test_completed_stage_is_not_reported_as_stalled(prepared):
    import json
    import os

    from typer.testing import CliRunner

    from preanchor.cli import app

    directory, config, _ = prepared
    run_smoke(directory, ROOT, config, tracking=False)
    os.utime(directory / 'heartbeat.json', (0, 0))
    result = CliRunner().invoke(app, ['status', '--root', str(directory.parents[2])])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['stale'] is False


def test_main_loop_heartbeat_updates_at_30_seconds(tmp_path, monkeypatch):
    from preanchor.runtime import Runtime

    clock = [0.0]
    monkeypatch.setattr('preanchor.runtime.time.monotonic', lambda: clock[0])
    runtime = Runtime(tmp_path, 'fixture', 1)
    runtime.pulse(force=True)
    heartbeat = tmp_path / 'heartbeat.json'
    original = heartbeat.read_bytes()
    clock[0] = 29
    runtime.pulse()
    assert heartbeat.read_bytes() == original
    clock[0] = 30
    runtime.pulse()
    assert heartbeat.read_bytes() != original


def test_wandb_online_failure_falls_back_without_credentials_in_config(tmp_path, monkeypatch):
    import wandb

    calls = []
    monkeypatch.setenv('WANDB_API_KEY', 'test-secret-value')
    monkeypatch.setenv('WANDB_MODE', 'online')
    def initialize(**kwargs):
        calls.append(kwargs)
        if kwargs['mode'] == 'online':
            raise ConnectionError('simulated outage')
        return SimpleNamespace(id='offline-fixture', dir=str(tmp_path / 'wandb/files'))
    monkeypatch.setattr(wandb, 'init', initialize)
    config = load_config(ROOT / 'configs/smoke.yaml')
    session, record = start_tracking(tmp_path, {'run_id': 'fixture', 'identity': {}}, config)
    assert session.id == 'offline-fixture'
    assert record['mode'] == 'offline'
    assert [call['mode'] for call in calls] == ['online', 'offline']
    assert 'test-secret-value' not in str(calls)
