import os

import pytest

from detllm import api


pytestmark = [
    pytest.mark.integration,
]


def test_hf_check_e2e(tmp_path):
    report = api.check(
        backend="hf",
        model="sshleifer/tiny-gpt2",
        prompts=["Hello"],
        runs=2,
        out_dir=str(tmp_path / "out"),
    )
    assert report.status in {"PASS", "FAIL"}


def test_hf_tier2_scores_present(tmp_path):
    report = api.check(
        backend="hf",
        model="sshleifer/tiny-gpt2",
        prompts=["Hello"],
        runs=1,
        tier=2,
        out_dir=str(tmp_path / "out_t2"),
    )
    assert report.status in {"PASS", "FAIL"}
    trace_path = tmp_path / "out_t2" / "traces" / "run_0.jsonl"
    content = trace_path.read_text(encoding="utf-8").strip()
    assert "\"scores\"" in content


def test_hf_check_second_model(tmp_path):
    report = api.check(
        backend="hf",
        model="hf-internal-testing/tiny-random-gpt2",
        prompts=["Hello"],
        runs=1,
        out_dir=str(tmp_path / "out_tiny_random"),
    )
    assert report.status in {"PASS", "FAIL"}
