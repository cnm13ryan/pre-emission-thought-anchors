"""Explicit YAML resolution and environment-only operational settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently changing a frozen choice."""


def _mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in ancestors:
        raise ValueError("Configuration inheritance cycle")
    value = read_yaml(path)
    parent = value.pop("extends", None)
    if parent is not None:
        value = merge(_resolve(path.parent / parent, (*ancestors, path)), value)
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dtype: Literal["bfloat16"]
    quantization: None
    context_length: Literal[32768]
    attention_implementation: Literal["sdpa"]
    decoder_layers: Literal[48]
    d_model: Literal[5120]
    chat_template: None


class DatasetConfig(StrictModel):
    id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    problem_ids: list[int]
    trace_kinds: list[Literal["correct", "incorrect"]]
    split_manifest: str


class Splits(StrictModel):
    schema_version: Literal[1]
    train: list[int]
    validation: list[int]
    test: list[int]

    @model_validator(mode="after")
    def disjoint(self) -> Splits:
        groups = [self.train, self.validation, self.test]
        flat = [problem for group in groups for problem in group]
        if any(not group for group in groups) or len(flat) != len(set(flat)):
            raise ValueError("Problem splits must be nonempty, unique and mutually disjoint")
        return self


class TrackingConfig(StrictModel):
    wandb_project: str
    wandb_mode: Literal["online", "offline"]
    streamlit_address: Literal["127.0.0.1"]
    streamlit_port: int = Field(ge=1024, le=65535)


class ExecutionConfig(StrictModel):
    mode: Literal["mvp", "smoke", "pilot"]
    extraction_batch_size: Literal[1]
    heartbeat_seconds: Literal[30]
    backup_interval_seconds: Literal[900]
    trajectory: str | None = None
    limit_sentences: int | None = Field(default=None, gt=0)


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    seed_root: int = Field(ge=0, le=2**32 - 1)
    model: ModelConfig
    dataset: DatasetConfig
    segmentation: dict[str, Any]
    labels: dict[str, Any]
    features: dict[str, Any]
    probes: dict[str, Any]
    evaluation: dict[str, Any]
    generation: dict[str, Any]
    tracking: TrackingConfig
    execution: ExecutionConfig
    splits: Splits

    @model_validator(mode="after")
    def scientific_contract(self) -> ExperimentConfig:
        ids = self.splits.train + self.splits.validation + self.splits.test
        if len(self.dataset.problem_ids) != 20 or set(ids) != set(self.dataset.problem_ids):
            raise ValueError("Dataset and split must contain the same 20 distinct problems")
        if len(ids) != 20 or list(map(len, [self.splits.train, self.splits.validation,
                                          self.splits.test])) != [12, 4, 4]:
            raise ValueError("MVP requires 12/4/4 problem splits")
        if sorted(self.dataset.trace_kinds) != ["correct", "incorrect"]:
            raise ValueError("Both trajectory kinds are required")
        if self.features.get("layers") != list(range(49)):
            raise ValueError("All 49 activation sites are required")
        if self.features.get("prohibit_future_features") is not True:
            raise ValueError("Future features must be prohibited")
        if self.evaluation.get("test_locked") is not True:
            raise ValueError("Test evaluation must be locked")
        if self.probes.get("group_column") != "problem_id":
            raise ValueError("CV must group by problem_id")
        return self

    def resolved_hash(self) -> str:
        return config_hash(self.model_dump(mode="json"))

    def scientific_hash(self) -> str:
        # Display/tracking changes must not invalidate measured artifacts (SPEC §10.4).
        return config_hash(self.model_dump(mode="json", exclude={"tracking"}))


def config_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_config(path: Path, *, repo_root: Path | None = None) -> ExperimentConfig:
    value = _resolve(path)
    root = repo_root or Path(__file__).resolve().parents[2]
    split_path = root / value["dataset"]["split_manifest"]
    value["splits"] = read_yaml(split_path)
    return ExperimentConfig.model_validate(value)


class Settings(BaseSettings):
    """No credentials here: backend clients read secrets directly from the environment."""

    model_config = SettingsConfigDict(env_prefix="PREANCHOR_", extra="ignore")
    root: Path | None = None
    artifacts: Path | None = None

    def persistent_root(self) -> Path:
        if self.root is None:
            raise ValueError("PREANCHOR_ROOT must be explicitly set")
        root = self.root.resolve()
        if not root.is_relative_to(Path("/lambda/nfs")) or root == Path("/lambda/nfs"):
            raise ValueError("PREANCHOR_ROOT must resolve beneath /lambda/nfs/")
        if self.artifacts is not None and self.artifacts.resolve() != root / "artifacts":
            raise ValueError("PREANCHOR_ARTIFACTS must be PREANCHOR_ROOT/artifacts")
        return root
