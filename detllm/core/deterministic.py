"""Determinism controls and context manager."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
import os
import random
from typing import Any


@dataclass
class DeterminismApplied:
    tier_requested: int
    tier_effective: int
    mode: str
    seed: int | None
    seed_controls: dict[str, Any] = field(default_factory=dict)
    torch_controls: dict[str, Any] = field(default_factory=dict)
    env_controls: dict[str, Any] = field(default_factory=dict)
    downgrades: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    capability_failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicContext(AbstractContextManager):
    def __init__(self, tier: int, mode: str, seed: int | None):
        self.tier = tier
        self.mode = mode
        self.seed = seed
        self.applied = DeterminismApplied(
            tier_requested=tier, tier_effective=tier, mode=mode, seed=seed
        )
        self._rng_state = None
        self._torch_state = None

    def __enter__(self):
        self._rng_state = random.getstate()
        random.seed(self.seed or 0)
        self.applied.seed_controls["python_random"] = True

        try:
            import torch

            self._torch_state = torch.random.get_rng_state()
            if self.seed is not None:
                torch.manual_seed(self.seed)
                self.applied.seed_controls["torch_manual_seed"] = True

            torch.use_deterministic_algorithms(True)
            self.applied.torch_controls["use_deterministic_algorithms"] = True

            self.applied.env_controls["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            )

        except Exception as exc:
            if self.mode == "strict":
                raise
            self.applied.downgrades.append(
                {
                    "from": self.tier,
                    "to": 0,
                    "reason": f"failed to apply torch controls: {exc}",
                }
            )
            self.applied.tier_effective = 0
            self.applied.warnings.append(str(exc))

        return self

    def __exit__(self, exc_type, exc, tb):
        random.setstate(self._rng_state)
        try:
            import torch

            if self._torch_state is not None:
                torch.random.set_rng_state(self._torch_state)
        except Exception:
            pass
        return False
