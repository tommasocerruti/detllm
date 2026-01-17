import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "detllm.cli.main", "-h"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Deterministic-mode checks" in result.stdout
