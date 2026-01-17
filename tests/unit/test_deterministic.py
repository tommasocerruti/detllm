import builtins
import pytest

from detllm.core.deterministic import DeterministicContext


def test_deterministic_context_best_effort_without_torch(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with DeterministicContext(tier=1, mode="best-effort", seed=123) as ctx:
        assert ctx.applied.tier_effective == 0
        assert ctx.applied.downgrades


def test_deterministic_context_strict_requires_torch(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        with DeterministicContext(tier=1, mode="strict", seed=123):
            pass
