"""Artifact helpers for DetLLM."""

from __future__ import annotations

from typing import Any

import importlib.resources as resources

REQUIRED_HEADER_FIELDS = {"schema_version", "detllm_version", "artifact_type"}
SCHEMA_MAP = {
    "env_snapshot": "env_snapshot",
    "run_config": "run_config",
    "determinism_applied": "determinism_applied",
    "report": "report",
}


def load_json(path: str) -> dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = REQUIRED_HEADER_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required header fields: {sorted(missing)}")

    # TODO: Validate against a JSON schema once the artifact models stabilize.
    return data


def dump_json(path: str, data: dict[str, Any]) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_schema(name: str) -> dict[str, Any]:
    import json

    schema_path = f"{name}.json"
    with resources.files("detllm.schemas").joinpath(schema_path).open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def validate_json(data: dict[str, Any], schema: dict[str, Any] | None) -> None:
    if schema is None:
        return
    try:
        import jsonschema
    except Exception as exc:
        raise RuntimeError("jsonschema is required for schema validation") from exc
    jsonschema.validate(instance=data, schema=schema)


def validate_artifact(data: dict[str, Any]) -> None:
    artifact_type = data.get("artifact_type")
    schema_name = SCHEMA_MAP.get(artifact_type)
    if not schema_name:
        return
    validate_json(data, load_schema(schema_name))
