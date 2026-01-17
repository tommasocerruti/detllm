"""Artifact helpers for DetLLM."""

from __future__ import annotations

from typing import Any

REQUIRED_HEADER_FIELDS = {"schema_version", "detllm_version", "artifact_type"}


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
