"""Core artifact models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvSnapshot:
    schema_version: str
    detllm_version: str
    artifact_type: str
    python: dict[str, Any]
    platform: dict[str, Any]
    torch: dict[str, Any]
    transformers: dict[str, Any]
    device: dict[str, Any] | None
    env_vars: dict[str, Any]
    fingerprint: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvSnapshot":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class RunConfig:
    schema_version: str
    detllm_version: str
    artifact_type: str
    backend: str
    tier_requested: int
    tier_effective: int
    mode: str
    model: str
    dtype: str
    device: str
    decoding: dict[str, Any]
    batch_size: int
    vary_batch: list[int]
    tokenizer: dict[str, Any]
    generation_context: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class DeterminismAppliedRecord:
    schema_version: str
    detllm_version: str
    artifact_type: str
    tier_requested: int
    tier_effective: int
    mode: str
    seed: int | None
    seed_controls: dict[str, Any]
    torch_controls: dict[str, Any]
    env_controls: dict[str, Any]
    downgrades: list[dict[str, Any]]
    warnings: list[str]
    capability_failures: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeterminismAppliedRecord":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class TokenTraceRow:
    prompt_id: str
    input_token_ids: list[int]
    input_token_ids_hash: str
    generated_token_ids: list[int]
    scores: list[float] | None
    tokenizer_id: str | None = None
    decoding_max_new_tokens: int | None = None
    decoding_do_sample: bool | None = None
    decoding_temperature: float | None = None
    decoding_top_p: float | None = None
    decoding_top_k: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenTraceRow":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
