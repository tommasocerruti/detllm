"""Render adaptive experiment plan artifacts."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detllm.experiment_planner.planner import ExperimentPlan


def write_plan_csv(plan: ExperimentPlan, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    axis_names = sorted(
        {
            str(axis)
            for item in plan.recommendations
            for axis in item.axes.keys()
        }
    )
    fieldnames = [
        "rank",
        "recommendation_type",
        "score",
        "reason",
        "source_cell_ids",
        "suggested_overrides",
        "command_hint",
    ] + [f"axis_{name}" for name in axis_names]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in plan.recommendations:
            row = {
                "rank": item.rank,
                "recommendation_type": item.recommendation_type,
                "score": item.score,
                "reason": item.reason,
                "source_cell_ids": ",".join(item.source_cell_ids),
                "suggested_overrides": item.suggested_overrides,
                "command_hint": item.command_hint,
            }
            row.update({f"axis_{name}": item.axes.get(name, "") for name in axis_names})
            writer.writerow(row)


def render_plan_text(plan: ExperimentPlan) -> str:
    lines = [
        "Adaptive Experiment Plan",
        f"Strategy: {plan.strategy}",
        f"Budget cells: {plan.budget_cells}",
        f"Recommendations: {len(plan.recommendations)}",
    ]
    if plan.analysis_summary:
        lines.extend(
            [
                "",
                "Analysis summary:",
                f"- Executed: {plan.analysis_summary.get('executed', 0)}",
                f"- Stable: {plan.analysis_summary.get('stable', 0)}",
                f"- Fragile: {plan.analysis_summary.get('fragile', 0)}",
                f"- Unstable: {plan.analysis_summary.get('unstable', 0)}",
                f"- Skipped: {plan.analysis_summary.get('skipped', 0)}",
            ]
        )
    if plan.recommendations:
        lines.extend(["", "Top recommendations:"])
        for item in plan.recommendations:
            lines.append(
                f"{item.rank}. {item.recommendation_type} "
                f"(score {item.score:.2f}): {item.reason}"
            )
            lines.append(f"   {item.command_hint}")
    return "\n".join(lines) + "\n"
