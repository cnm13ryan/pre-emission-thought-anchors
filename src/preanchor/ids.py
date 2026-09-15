"""Content identities and deterministic seeds (SPEC §3.13 and §9)."""

import hashlib
from pathlib import Path

from preanchor.config import config_hash


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_seed(seed_root: int, trajectory_id: str, fork_idx: int, sample_idx: int) -> int:
    value = f"{seed_root}:{trajectory_id}:{fork_idx}:{sample_idx}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def derived_seed(seed_root: int, purpose: str, *parts: object) -> int:
    """Namespaced seeds for folds, bootstrap and qualitative sampling."""
    return int(config_hash([seed_root, purpose, *parts])[:8], 16)


def cache_key(*, stage: str, scientific_config_hash: str, model_revision: str,
              tokenizer_revision: str, trace_sha256: str | None,
              segmentation_version: str, boundary_mapping_sha256: str | None,
              generation_parameters: dict, fork_idx: int | None, sample_seed: int | None,
              code_version: str, input_hashes: dict[str, str], unit_id: str) -> str:
    """Missing stage-specific inputs must be explicit nulls, never implicit omissions."""
    return config_hash(locals())
