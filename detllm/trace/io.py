"""Trace IO helpers."""

from __future__ import annotations

import json
from typing import Any, Iterable


def write_trace(path: str, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
