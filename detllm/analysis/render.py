"""Render statistical analysis artifacts."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detllm.analysis.analysis import AnalysisResult


def write_analysis_csv(result: AnalysisResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    axis_names = sorted(
        {
            str(axis)
            for cell in result.cells
            for axis in cell.get("axes", {}).keys()
        }
    )
    fieldnames = [
        "cell_id",
        "execution_status",
        "classification",
        "risk_weight",
        "report_status",
        "report_category",
        "min_topk_margin",
        "artifact_path",
    ] + [f"axis_{name}" for name in axis_names]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in result.cells:
            row = {
                "cell_id": cell.get("cell_id", ""),
                "execution_status": cell.get("execution_status", ""),
                "classification": cell.get("classification") or "",
                "risk_weight": _empty_if_none(cell.get("risk_weight")),
                "report_status": cell.get("report_status") or "",
                "report_category": cell.get("report_category") or "",
                "min_topk_margin": _empty_if_none(cell.get("min_topk_margin")),
                "artifact_path": cell.get("artifact_path") or "",
            }
            axes = cell.get("axes", {})
            row.update({f"axis_{name}": axes.get(name, "") for name in axis_names})
            writer.writerow(row)


def render_analysis_text(result: AnalysisResult) -> str:
    summary = result.summary
    risk = "unknown" if result.risk_score is None else f"{result.risk_score:.2f}"
    unstable = result.intervals["unstable_rate"]
    nonstable = result.intervals["nonstable_rate"]
    lines = [
        "Statistical Reproducibility Analysis",
        f"Executed cells: {summary['executed']}",
        f"Stable: {summary['stable']}",
        f"Fragile: {summary['fragile']}",
        f"Unstable: {summary['unstable']}",
        f"Skipped: {summary['skipped']}",
        f"Reproducibility risk score: {risk}",
        _format_interval("Unstable rate", unstable),
        _format_interval("Non-stable rate", nonstable),
    ]
    if summary.get("rule_of_three"):
        lines.extend(["", summary["rule_of_three"]])
    if result.recommendations:
        lines.extend(["", "Recommendations:"])
        for item in result.recommendations:
            lines.append(f"- {item['code']}: {item['summary']}")
    return "\n".join(lines) + "\n"


def _format_interval(label: str, interval: dict[str, object]) -> str:
    if interval["rate"] is None:
        return f"{label}: unknown"
    confidence = int(round(float(interval["confidence"]) * 100))
    return (
        f"{label}: {float(interval['rate']):.3f} "
        f"({confidence}% CI {float(interval['lower']):.3f}-{float(interval['upper']):.3f})"
    )


def _empty_if_none(value: object) -> object:
    if value is None:
        return ""
    return value
