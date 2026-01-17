"""Capability checks for determinism tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from detllm.backends.base import BackendCapabilities
from detllm.core.deterministic import DeterminismApplied


@dataclass(frozen=True)
class CapabilityDecision:
    supported: bool
    tier_effective: int
    capability_failures: list[dict[str, Any]]
    notes: list[str]


def evaluate_capabilities(
    applied: DeterminismApplied,
    capabilities: BackendCapabilities,
    tier_requested: int,
    mode: str,
) -> CapabilityDecision:
    failures: list[dict[str, Any]] = []
    notes = list(capabilities.notes)
    tier_effective = tier_requested

    if tier_requested >= 1 and not capabilities.supports_torch_deterministic:
        failures.append(
            {
                "requirement": "torch_deterministic",
                "reason": "backend does not support deterministic torch controls",
            }
        )
        tier_effective = min(tier_effective, 0)

    if tier_requested >= 1 and not capabilities.supports_tier1_fixed_batch:
        failures.append(
            {
                "requirement": "tier1_fixed_batch",
                "reason": "backend does not support fixed-batch repeatability",
            }
        )
        tier_effective = min(tier_effective, 0)

    if tier_requested >= 2 and not capabilities.supports_scores:
        failures.append(
            {
                "requirement": "scores",
                "reason": "backend does not support score/logprob capture",
            }
        )
        tier_effective = min(tier_effective, 1)

    if failures:
        applied.capability_failures.extend(failures)

        if mode == "strict":
            return CapabilityDecision(
                supported=False,
                tier_effective=tier_effective,
                capability_failures=failures,
                notes=notes,
            )

        if tier_effective < tier_requested:
            applied.downgrades.append(
                {
                    "from": tier_requested,
                    "to": tier_effective,
                    "reason": "capability limits",
                }
            )
            applied.tier_effective = tier_effective

    return CapabilityDecision(
        supported=True,
        tier_effective=applied.tier_effective,
        capability_failures=failures,
        notes=notes,
    )
