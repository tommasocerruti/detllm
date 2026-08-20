"""Offline setup checks for detLLM."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

from detllm.usability.config import load_project_config


def run_doctor(
    *,
    config_path: str = "detllm.config.json",
    backend: str | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(_python_version_check())
    config = None
    try:
        config = load_project_config(config_path)
        checks.append(
            _check(
                "config_readable",
                "PASS",
                f"Loaded project config from {config_path}",
                "No action needed.",
            )
        )
    except Exception as exc:
        checks.append(
            _check(
                "config_readable",
                "FAIL",
                f"Could not load project config: {exc}",
                "Run detllm init --out . or pass --config PATH.",
            )
        )

    artifact_root = config.get("artifact_root") if config else "artifacts"
    checks.append(_writable_artifact_root_check(str(artifact_root)))
    checks.append(_import_check("jsonschema", "schema_extra", "Install detllm[schema]."))
    if backend in {None, "hf"}:
        checks.extend(
            [
                _import_check("torch", "backend_hf_extra", "Install detllm[hf]."),
                _import_check("transformers", "backend_hf_extra", "Install detllm[hf]."),
            ]
        )
    if backend in {None, "vllm"}:
        checks.append(_import_check("vllm", "backend_vllm_extra", "Install detllm[vllm]."))

    return {
        "status": _overall_status(checks),
        "checks": checks,
    }


def _python_version_check() -> dict[str, Any]:
    version = sys.version_info
    if version >= (3, 10):
        return _check(
            "python_version",
            "PASS",
            f"Python {version.major}.{version.minor}.{version.micro} is supported.",
            "No action needed.",
        )
    return _check(
        "python_version",
        "FAIL",
        f"Python {version.major}.{version.minor}.{version.micro} is unsupported.",
        "Use Python 3.10 or newer.",
    )


def _writable_artifact_root_check(path: str) -> dict[str, Any]:
    parent = _nearest_existing_parent(path)
    if os.access(parent, os.W_OK):
        return _check(
            "artifact_root_writable",
            "PASS",
            f"Nearest existing artifact root parent is writable: {parent}",
            "No action needed.",
        )
    return _check(
        "artifact_root_writable",
        "FAIL",
        f"Nearest existing artifact root parent is not writable: {parent}",
        "Choose a writable artifact_root in detllm.config.json.",
    )


def _nearest_existing_parent(path: str) -> str:
    candidate = os.path.abspath(path)
    if not os.path.isdir(candidate):
        candidate = os.path.dirname(candidate) or "."

    while not os.path.exists(candidate):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent

    return candidate


def _import_check(module: str, code: str, remediation: str) -> dict[str, Any]:
    if importlib.util.find_spec(module) is not None:
        return _check(code, "PASS", f"Optional module is available: {module}", "No action needed.")
    return _check(code, "WARN", f"Optional module is missing: {module}", remediation)


def _check(code: str, status: str, summary: str, remediation: str) -> dict[str, Any]:
    return {
        "code": code,
        "status": status,
        "summary": summary,
        "remediation": remediation,
    }


def _overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {check["status"] for check in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"
