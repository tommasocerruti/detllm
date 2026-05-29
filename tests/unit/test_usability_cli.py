import json
import subprocess
import sys


def test_init_cli_writes_expected_files(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "init",
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    assert (tmp_path / "detllm.config.json").exists()
    assert (tmp_path / "prompts.jsonl").exists()


def test_doctor_cli_json_succeeds(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "init",
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "doctor",
            "--config",
            str(tmp_path / "detllm.config.json"),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] in {"PASS", "WARN"}


def test_inspect_cli_phase_fixture_succeeds():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "inspect",
            "--in",
            "examples/fixtures/phase_demo",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "phase_diagram" in result.stdout


def test_profile_list_and_dry_run_cli(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "init",
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    list_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "profile",
            "list",
            "--config",
            str(tmp_path / "detllm.config.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dry_run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "profile",
            "run",
            "analyze_fixture",
            "--config",
            str(tmp_path / "detllm.config.json"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "quick_check" in list_result.stdout
    assert "detllm analyze --in examples/fixtures/phase_demo" in dry_run_result.stdout
