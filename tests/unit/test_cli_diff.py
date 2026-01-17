import json
import subprocess
import sys

from detllm.trace.io import write_trace


def test_cli_diff_writes_report(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    out_dir = tmp_path / "out"

    write_trace(
        str(left),
        [{"prompt_id": "a", "generated_token_ids": [1, 2]}],
    )
    write_trace(
        str(right),
        [{"prompt_id": "a", "generated_token_ids": [1, 2]}],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "diff",
            "--left",
            str(left),
            "--right",
            str(right),
            "--out",
            str(out_dir),
            "--report",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""
    assert "Status: PASS" in result.stdout

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["category"] == "PASS"
