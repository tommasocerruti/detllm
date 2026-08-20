"""Project config support for detLLM onboarding commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_PROFILE_COMMANDS = {
    "check",
    "phase",
    "analyze",
    "recommend",
    "diagnose",
    "replay",
}


def starter_config(artifact_root: str = "artifacts/detllm") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_root": artifact_root,
        "profiles": {
            "quick_check": {
                "command": "check",
                "options": {
                    "backend": "hf",
                    "model": "distilgpt2",
                    "prompt_file": "prompts.jsonl",
                    "tier": 1,
                    "runs": 3,
                    "batch_size": 1,
                    "out": f"{artifact_root}/quick_check",
                },
            },
            "replayable_check": {
                "command": "check",
                "options": {
                    "backend": "hf",
                    "model": "distilgpt2",
                    "prompt_file": "prompts.jsonl",
                    "tier": 2,
                    "runs": 3,
                    "batch_size": 1,
                    "vary_batch": "1,2",
                    "capture_topk_scores": 5,
                    "include_token_text": True,
                    "out": f"{artifact_root}/replayable_check",
                },
            },
            "phase_demo": {
                "command": "phase",
                "options": {
                    "backend": "hf",
                    "model": "distilgpt2",
                    "prompt_file": "prompts.jsonl",
                    "axis": [
                        "batch_size=1,2",
                        "dtype=float32",
                        "max_new_tokens=16,32",
                    ],
                    "runs": 3,
                    "tier": 2,
                    "capture_topk_scores": 5,
                    "max_cells": 4,
                    "out": f"{artifact_root}/phase_demo",
                },
            },
            "analyze_fixture": {
                "command": "analyze",
                "options": {
                    "in": "examples/fixtures/phase_demo",
                    "out": f"{artifact_root}/phase_fixture/analysis",
                },
            },
            "recommend_fixture": {
                "command": "recommend",
                "options": {
                    "in": "examples/fixtures/phase_demo",
                    "out": f"{artifact_root}/phase_fixture/recommendations",
                    "budget_cells": 8,
                    "strategy": "auto",
                },
            },
        },
    }


def load_project_config(path: str = "detllm.config.json") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_project_config(config)
    return config


def validate_project_config(config: dict[str, Any]) -> None:
    if "schema_version" not in config:
        raise ValueError("Project config must include schema_version")
    if "artifact_root" not in config:
        raise ValueError("Project config must include artifact_root")
    if "profiles" not in config or not isinstance(config["profiles"], dict):
        raise ValueError("Project config must include profiles")

    for name, profile in config["profiles"].items():
        if not isinstance(profile, dict):
            raise ValueError(f"Profile {name} must be an object")
        command = profile.get("command")
        if command not in SUPPORTED_PROFILE_COMMANDS:
            raise ValueError(f"Unsupported profile command: {command}")
        options = profile.get("options")
        if not isinstance(options, dict):
            raise ValueError(f"Profile {name} must include options")


def render_profile_command(config: dict[str, Any], name: str) -> list[str]:
    validate_project_config(config)
    if name not in config["profiles"]:
        raise KeyError(f"Unknown profile: {name}")
    profile = config["profiles"][name]
    return ["detllm", profile["command"], *_render_options(profile["options"])]


def profile_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_project_config(config)
    rows = []
    for name, profile in sorted(config["profiles"].items()):
        options = profile["options"]
        rows.append(
            {
                "name": name,
                "command": profile["command"],
                "backend": options.get("backend", ""),
                "model": options.get("model", ""),
                "out": options.get("out", ""),
            }
        )
    return rows


def config_path_in(directory: str) -> Path:
    return Path(directory) / "detllm.config.json"


def prompts_path_in(directory: str) -> Path:
    return Path(directory) / "prompts.jsonl"


def _render_options(options: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    for key in sorted(options):
        value = options[key]
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                rendered.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                rendered.extend([flag, str(item)])
            continue
        rendered.extend([flag, str(value)])
    return rendered
