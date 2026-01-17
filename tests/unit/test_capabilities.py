from detllm.backends.base import BackendCapabilities
from detllm.core.capabilities import evaluate_capabilities
from detllm.core.deterministic import DeterminismApplied


def test_capability_strict_rejects_missing_scores():
    applied = DeterminismApplied(tier_requested=2, tier_effective=2, mode="strict", seed=0)
    caps = BackendCapabilities(
        supports_tier1_fixed_batch=True,
        supports_scores=False,
        supports_torch_deterministic=True,
        notes=["score support missing"],
    )
    decision = evaluate_capabilities(applied, caps, tier_requested=2, mode="strict")
    assert decision.supported is False
    assert decision.capability_failures
    assert decision.notes == ["score support missing"]


def test_capability_best_effort_downgrades():
    applied = DeterminismApplied(tier_requested=2, tier_effective=2, mode="best-effort", seed=0)
    caps = BackendCapabilities(
        supports_tier1_fixed_batch=True,
        supports_scores=False,
        supports_torch_deterministic=True,
    )
    decision = evaluate_capabilities(applied, caps, tier_requested=2, mode="best-effort")
    assert decision.supported is True
    assert applied.tier_effective == 1
    assert applied.downgrades
