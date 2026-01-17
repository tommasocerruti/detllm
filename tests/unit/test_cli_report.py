import subprocess
import sys


def test_cli_report_renders_text(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        """
        {
          "schema_version": "1.0",
          "detllm_version": "0.1.0",
          "artifact_type": "report",
          "status": "PASS",
          "category": "PASS",
          "details": {"runs": 1}
        }
        """,
        encoding="utf-8",
    )
    out_path = tmp_path / "report.txt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "report",
            "--in",
            str(report_path),
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""
    content = out_path.read_text(encoding="utf-8")
    assert "Status: PASS" in content
