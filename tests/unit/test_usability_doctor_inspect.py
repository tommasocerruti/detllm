import json

import pytest

from detllm.usability.doctor import run_doctor
from detllm.usability.init import init_project
from detllm.usability.inspect import inspect_artifact_dir


def test_doctor_reports_status_and_checks(tmp_path):
    init_project(str(tmp_path))

    report = run_doctor(config_path=str(tmp_path / "detllm.config.json"))

    assert report["status"] in {"PASS", "WARN"}
    assert report["checks"]
    assert any(check["code"] == "python_version" for check in report["checks"])
    assert any(check["code"] == "config_readable" for check in report["checks"])


def test_doctor_missing_config_is_clear_failure(tmp_path):
    report = run_doctor(config_path=str(tmp_path / "missing.json"))

    assert report["status"] == "FAIL"
    assert any(check["code"] == "config_readable" for check in report["checks"])


def test_doctor_json_payload_is_serializable(tmp_path):
    init_project(str(tmp_path))

    report = run_doctor(config_path=str(tmp_path / "detllm.config.json"), backend="hf")

    json.dumps(report)
    assert any(check["code"] == "backend_hf_extra" for check in report["checks"])


def test_inspect_phase_fixture_summary():
    summary = inspect_artifact_dir("examples/fixtures/phase_demo")

    assert summary["artifact_type"] == "phase_diagram"
    assert summary["status"] == "PASS"
    assert summary["summary"]["stable"] == 1
    assert summary["next_command"] == "detllm analyze --in examples/fixtures/phase_demo"


def test_inspect_analysis_summary(tmp_path):
    from detllm import analyze

    analyze(
        "examples/fixtures/phase_demo",
        out_dir=str(tmp_path / "analysis"),
    )

    summary = inspect_artifact_dir(str(tmp_path / "analysis"))

    assert summary["artifact_type"] == "analysis"
    assert summary["summary"]["unstable"] == 2
    assert summary["next_command"].startswith("detllm recommend --in")


def test_inspect_recommendation_summary(tmp_path):
    from detllm import recommend

    recommend(
        "examples/fixtures/phase_demo",
        out_dir=str(tmp_path / "recommendations"),
    )

    summary = inspect_artifact_dir(str(tmp_path / "recommendations"))

    assert summary["artifact_type"] == "experiment_plan"
    assert summary["recommendation_count"] > 0
    assert summary["status"] == "PASS"


def test_inspect_unknown_directory_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No known detLLM artifacts"):
        inspect_artifact_dir(str(tmp_path))
