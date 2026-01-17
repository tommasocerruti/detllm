"""Render report to text."""

from __future__ import annotations

from detllm.report.report import Report


def render_report(report: Report) -> str:
    lines = [
        f"Status: {report.status}",
        f"Category: {report.category}",
    ]
    if report.details:
        lines.append("Details:")
        for key, value in report.details.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"
