import os

import pytest

from detllm import api


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("DETLLM_RUN_INTEGRATION") != "1",
        reason="integration test disabled",
    ),
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
