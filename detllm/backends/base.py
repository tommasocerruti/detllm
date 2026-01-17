"""Backend interface definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    supports_tier1_fixed_batch: bool
    supports_scores: bool
    supports_torch_deterministic: bool
    notes: list[str] = field(default_factory=list)


class BackendAdapter(Protocol):
    def capabilities(self) -> BackendCapabilities:
        ...

    def generate(self, prompts: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        ...
