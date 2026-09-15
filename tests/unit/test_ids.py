import hashlib

from preanchor.ids import cache_key, derived_seed, sample_seed


def test_sample_seed_matches_spec_exactly():
    expected = int(hashlib.sha256(b'20260914:4682:correct:12:3').hexdigest()[:8], 16)
    assert sample_seed(20260914, '4682:correct', 12, 3) == expected
    assert 0 <= expected < 2**32
    assert derived_seed(1, 'fold', 0) != derived_seed(1, 'bootstrap', 0)


def test_cache_identity_covers_every_result_affecting_input():
    fields = dict(stage='smoke', scientific_config_hash='science', model_revision='model',
                  tokenizer_revision='tokenizer', trace_sha256='trace',
                  segmentation_version='released', boundary_mapping_sha256='boundary',
                  generation_parameters={'temperature': 0.6}, fork_idx=1, sample_seed=2,
                  code_version='code', input_hashes={'file': 'hash'}, unit_id='unit')
    original = cache_key(**fields)
    for key in fields:
        changed = dict(fields)
        changed[key] = 'changed'
        assert cache_key(**changed) != original
    assert cache_key(**dict(reversed(list(fields.items())))) == original
