import pytest

from detllm.core.artifacts import load_schema
from detllm.usability.config import (
    load_project_config,
    render_profile_command,
    validate_project_config,
)
from detllm.usability.init import init_project


def test_init_creates_config_and_prompt_file(tmp_path):
    result = init_project(str(tmp_path))

    config_path = tmp_path / "detllm.config.json"
    prompt_path = tmp_path / "prompts.jsonl"
    config = load_project_config(str(config_path))

    assert result.status == "PASS"
    assert config_path.exists()
    assert prompt_path.exists()
    assert set(config["profiles"]) >= {
        "quick_check",
        "replayable_check",
        "phase_demo",
        "analyze_fixture",
        "recommend_fixture",
    }
    assert load_schema("project_config")["required"] == [
        "schema_version",
        "artifact_root",
        "profiles",
    ]
    validate_project_config(config)


def test_init_refuses_overwrite_without_force(tmp_path):
    init_project(str(tmp_path))

    with pytest.raises(FileExistsError, match="already exist"):
        init_project(str(tmp_path))


def test_init_force_overwrites_generated_files(tmp_path):
    init_project(str(tmp_path))
    (tmp_path / "prompts.jsonl").write_text("custom\n", encoding="utf-8")

    init_project(str(tmp_path), force=True)

    assert "custom" not in (tmp_path / "prompts.jsonl").read_text(encoding="utf-8")


def test_project_config_rejects_unknown_command(tmp_path):
    config = {
        "schema_version": "1.0",
        "artifact_root": "artifacts/detllm",
        "profiles": {
            "bad": {
                "command": "unknown",
                "options": {},
            }
        },
    }

    with pytest.raises(ValueError, match="Unsupported profile command"):
        validate_project_config(config)


def test_project_config_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="profiles"):
        validate_project_config({"schema_version": "1.0", "artifact_root": "artifacts"})


def test_render_profile_command_is_deterministic(tmp_path):
    init_project(str(tmp_path))
    config = load_project_config(str(tmp_path / "detllm.config.json"))

    command = render_profile_command(config, "analyze_fixture")

    assert command == [
        "detllm",
        "analyze",
        "--in",
        "examples/fixtures/phase_demo",
        "--out",
        "artifacts/detllm/phase_fixture/analysis",
    ]
