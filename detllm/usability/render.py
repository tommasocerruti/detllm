"""Text rendering for usability commands."""

from __future__ import annotations

from typing import Any


def render_init(result) -> str:
    lines = [
        "Initialized detLLM project",
        f"Config: {result.config_path}",
        f"Prompts: {result.prompt_path}",
        "",
        "Next commands:",
    ]
    lines.extend(f"- {command}" for command in result.next_commands)
    return "\n".join(lines) + "\n"


def render_doctor(report: dict[str, Any]) -> str:
    lines = ["detLLM Doctor", f"Status: {report['status']}", "", "Checks:"]
    for check in report["checks"]:
        lines.append(f"- {check['status']} {check['code']}: {check['summary']}")
        if check["status"] != "PASS":
            lines.append(f"  Fix: {check['remediation']}")
    return "\n".join(lines) + "\n"


def render_inspection(summary: dict[str, Any]) -> str:
    lines = [
        "detLLM Artifact Inspection",
        f"Type: {summary['artifact_type']}",
        f"Status: {summary.get('status', 'UNKNOWN')}",
    ]
    if summary.get("category"):
        lines.append(f"Category: {summary['category']}")
    if "summary" in summary:
        lines.append(f"Summary: {summary['summary']}")
    if "risk_score" in summary:
        lines.append(f"Risk score: {summary['risk_score']}")
    if "recommendation_count" in summary:
        lines.append(f"Recommendations: {summary['recommendation_count']}")
    if summary.get("first_divergence") is not None:
        lines.append(f"First divergence: {summary['first_divergence']}")
    if summary.get("next_command"):
        lines.extend(["", f"Next: {summary['next_command']}"])
    return "\n".join(lines) + "\n"


def render_profile_list(rows: list[dict[str, Any]]) -> str:
    lines = ["detLLM Profiles"]
    for row in rows:
        details = [row["command"]]
        if row.get("backend"):
            details.append(f"backend={row['backend']}")
        if row.get("model"):
            details.append(f"model={row['model']}")
        if row.get("out"):
            details.append(f"out={row['out']}")
        lines.append(f"- {row['name']}: " + ", ".join(details))
    return "\n".join(lines) + "\n"
