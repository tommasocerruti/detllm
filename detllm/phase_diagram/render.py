"""Render phase diagram artifacts."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detllm.phase_diagram.phase import PhaseDiagram


def write_phase_csv(diagram: PhaseDiagram, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cell_id",
            "execution_status",
            "classification",
            "report_status",
            "report_category",
            "min_topk_margin",
            "artifact_path",
        ]
        axis_names = sorted(diagram.axes)
        writer = csv.DictWriter(handle, fieldnames=fieldnames + axis_names)
        writer.writeheader()
        for cell in diagram.cells:
            row = {
                "cell_id": cell.cell_id,
                "execution_status": cell.execution_status,
                "classification": cell.classification or "",
                "report_status": cell.report_status or "",
                "report_category": cell.report_category or "",
                "min_topk_margin": cell.min_topk_margin
                if cell.min_topk_margin is not None
                else "",
                "artifact_path": cell.artifact_path or "",
            }
            row.update({name: cell.axes.get(name, "") for name in axis_names})
            writer.writerow(row)


def render_phase_text(diagram: PhaseDiagram) -> str:
    summary = diagram.summary()
    lines = [
        "Reproducibility Phase Diagram",
        f"Total cells: {summary['total']}",
        f"Stable: {summary['stable']}",
        f"Fragile: {summary['fragile']}",
        f"Unstable: {summary['unstable']}",
        f"Planned: {summary['planned']}",
        f"Skipped: {summary['skipped']}",
        "",
        "Cells:",
    ]
    for cell in diagram.cells:
        label = cell.classification or cell.execution_status
        lines.append(f"- {cell.cell_id}: {label}")
    return "\n".join(lines) + "\n"
