"""Trace IO helpers."""

from __future__ import annotations

import json
from typing import Any, Iterable

from detllm.core.artifacts import load_schema, validate_json


def write_trace(
    path: str,
    rows: Iterable[dict[str, Any]],
    validate_rows: bool = False,
) -> None:
    schema = load_schema("trace_row") if validate_rows else None
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            if schema is not None:
                validate_json(row, schema)
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def read_trace(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
