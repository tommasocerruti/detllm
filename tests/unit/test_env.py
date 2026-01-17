import json
import subprocess
import sys

from detllm.core import env as env_module


def test_capture_env_includes_header_and_fingerprint():
    snapshot = env_module.capture_env()
    for key in ("schema_version", "detllm_version", "artifact_type", "fingerprint"):
        assert key in snapshot

    fingerprint_payload = dict(snapshot)
    fingerprint_payload.pop("fingerprint")
    expected = env_module._canonical_fingerprint(fingerprint_payload)
    assert snapshot["fingerprint"] == expected


def test_cli_env_writes_file(tmp_path):
    out_path = tmp_path / "env.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "env",
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["artifact_type"] == "env_snapshot"
    assert "fingerprint" in data
