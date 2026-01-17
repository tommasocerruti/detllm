from dataclasses import dataclass

import pytest

from detllm import api
from detllm.backends.base import BackendCapabilities


@dataclass
class FakeBackend:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_tier1_fixed_batch=True,
            supports_scores=True,
            supports_torch_deterministic=True,
        )

    def generate(self, prompts, **kwargs):
        return [
            {"prompt": prompt, "input_ids": [1, 2], "output_ids": [3, 4], "scores": [0.1]}
            for prompt in prompts
        ]


def test_run_requires_prompts():
    with pytest.raises(ValueError):
        api.run(backend="hf", model="x", prompts=[])


def test_check_requires_prompts():
    with pytest.raises(ValueError):
        api.check(backend="hf", model="x", prompts=[])


def test_check_returns_report(tmp_path):
    report = api.check(
        backend="hf",
        model="fake",
        prompts=["hi"],
        runs=2,
        out_dir=str(tmp_path / "out"),
        backend_adapter=FakeBackend(),
    )
    assert report.status == "PASS"
