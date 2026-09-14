import hashlib
import json

import numpy as np
import pyarrow as pa
import pytest
from safetensors.numpy import load_file

from preanchor.io import (
    UnitStore,
    atomic_bytes,
    read_json,
    validate_parquet,
    write_json,
    write_jsonl,
    write_parquet,
    write_safetensors,
)

KEY = hashlib.sha256(b'unit').hexdigest()


def write_all(path):
    write_json(path / 'sample.json', {'schema_version': 1, 'value': 2})
    write_jsonl(path / 'events.jsonl', [{'step': 1}, {'step': 2}])
    write_parquet(path / 'sample.parquet', pa.table({'value': pa.array([2], type=pa.int32())}))
    write_safetensors(path / 'sample.safetensors', {'value': np.ones((2, 3), dtype=np.float16)})


def test_every_format_roundtrips_and_completed_units_are_immutable(tmp_path):
    store = UnitStore(tmp_path)
    path, created = store.produce(KEY, write_all)
    assert created
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
    assert read_json(path / 'sample.json')['value'] == 2
    assert validate_parquet(path / 'sample.parquet').to_pydict() == {'value': [2]}
    assert load_file(path / 'sample.safetensors')['value'].dtype == np.float16
    assert len((path / 'events.jsonl').read_text().splitlines()) == 2
    _, created = store.produce(KEY, lambda _: pytest.fail('Completed writer called'))
    assert not created
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}


def test_corruption_is_quarantined_and_only_that_unit_recomputed(tmp_path):
    store = UnitStore(tmp_path)
    path, _ = store.produce(KEY, write_all)
    valid = (path / 'sample.json').read_bytes()
    (path / 'sample.json').write_text('corrupted')
    store.produce(KEY, write_all)
    assert (path / 'sample.json').read_bytes() == valid
    quarantined = list((tmp_path / 'quarantine').glob('*/sample.json'))
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == 'corrupted'


def test_no_success_on_failed_writer_and_no_accepted_partial(tmp_path):
    def interrupted(path):
        write_json(path / 'sample.json', {'value': 1})
        raise RuntimeError('interrupted')
    store = UnitStore(tmp_path)
    with pytest.raises(RuntimeError):
        store.produce(KEY, interrupted)
    assert not (tmp_path / KEY / 'SUCCESS.json').exists()
    assert store.produce(KEY, write_all)[1]
    assert len(list((tmp_path / 'quarantine').iterdir())) == 1


def test_schema_corruption_is_detected(tmp_path):
    path = tmp_path / 'table.parquet'
    write_parquet(path, pa.table({'value': [1]}))
    schema_path = tmp_path / 'table.parquet.schema.json'
    value = json.loads(schema_path.read_text())
    value['row_count'] = 2
    payload = json.dumps(value).encode()
    schema_path.write_bytes(payload)
    (tmp_path / 'table.parquet.schema.json.sha256').write_text(hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match='schema'):
        validate_parquet(path)


def test_immutable_write_refuses_overwrite_and_symlink(tmp_path):
    path = tmp_path / 'file'
    atomic_bytes(path, b'original')
    with pytest.raises(FileExistsError):
        atomic_bytes(path, b'changed')
    link = tmp_path / 'link'
    link.symlink_to(path)
    with pytest.raises(ValueError):
        atomic_bytes(link, b'changed', replace=True)
    assert path.read_bytes() == b'original'
