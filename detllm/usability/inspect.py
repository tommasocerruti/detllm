"""Artifact directory inspection."""

from __future__ import annotations

import os
from typing import Any

from detllm.core.artifacts import load_json


def inspect_artifact_dir(path: str) -> dict[str, Any]:
    path = os.path.normpath(path)
    for filename, handler in _HANDLERS:
        artifact_path = os.path.join(path, filename)
        if os.path.exists(artifact_path):
            payload = load_json(artifact_path)
            return handler(path, payload)
    raise FileNotFoundError(f"No known detLLM artifacts found in {path}")


def _inspect_report(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    details = payload.get("details", {})
    return {
        "artifact_type": "report",
        "path": path,
        "status": payload.get("status", "UNKNOWN"),
        "category": payload.get("category"),
        "first_divergence": details.get("first_divergence"),
        "next_command": f"detllm diagnose --in {path}",
    }


def _inspect_diagnosis(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    causes = payload.get("causes", [])
    return {
        "artifact_type": "diagnosis",
        "path": path,
        "status": payload.get("status", "UNKNOWN"),
        "category": payload.get("category"),
        "cause_count": len(causes),
        "top_cause": causes[0].get("code") if causes else None,
        "next_command": f"detllm replay --in {path} --probe auto",
    }


def _inspect_replay(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": "replay",
        "path": path,
        "status": payload.get("status", "UNKNOWN"),
        "executed_probes": payload.get("executed_probes", []),
        "skipped_probes": payload.get("skipped_probes", []),
        "next_command": None,
    }


def _inspect_phase(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": "phase_diagram",
        "path": path,
        "status": "PASS",
        "summary": payload.get("summary", {}),
        "cell_count": len(payload.get("cells", [])),
        "next_command": f"detllm analyze --in {path}",
    }


def _inspect_analysis(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    phase_dir = os.path.dirname(path)
    return {
        "artifact_type": "analysis",
        "path": path,
        "status": "PASS",
        "summary": payload.get("summary", {}),
        "risk_score": payload.get("risk_score"),
        "next_command": f"detllm recommend --in {phase_dir}",
    }


def _inspect_plan(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": "experiment_plan",
        "path": path,
        "status": "PASS",
        "strategy": payload.get("strategy"),
        "recommendation_count": len(payload.get("recommendations", [])),
        "next_command": None,
    }


_HANDLERS: list[tuple[str, Any]] = [
    ("phase_diagram.json", _inspect_phase),
    ("analysis.json", _inspect_analysis),
    ("experiment_plan.json", _inspect_plan),
    ("diagnosis.json", _inspect_diagnosis),
    ("replay.json", _inspect_replay),
    ("report.json", _inspect_report),
]
