"""Reproducibility phase diagram models and grid logic."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from detllm.version import __version__

SUPPORTED_AXES = ("batch_size", "dtype", "max_new_tokens")


@dataclass
class PhaseCell:
    cell_id: str
    axes: dict[str, Any]
    execution_status: str = "planned"
    classification: str | None = None
    report_status: str | None = None
    report_category: str | None = None
    first_divergence: dict[str, Any] | None = None
    min_topk_margin: float | None = None
    diagnosis_causes: list[str] = field(default_factory=list)
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "axes": self.axes,
            "execution_status": self.execution_status,
            "classification": self.classification,
            "report_status": self.report_status,
            "report_category": self.report_category,
            "first_divergence": self.first_divergence,
            "min_topk_margin": self.min_topk_margin,
            "diagnosis_causes": self.diagnosis_causes,
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True)
class PhaseDiagram:
    axes: dict[str, list[Any]]
    cells: list[PhaseCell]
    out_dir: str
    dry_run: bool

    def summary(self) -> dict[str, int]:
        counts = {
            "stable": 0,
            "fragile": 0,
            "unstable": 0,
            "planned": 0,
            "skipped": 0,
        }
        for cell in self.cells:
            if cell.execution_status == "skipped":
                counts["skipped"] += 1
            elif cell.classification in {"stable", "fragile", "unstable"}:
                counts[cell.classification] += 1
            else:
                counts["planned"] += 1
        counts["total"] = len(self.cells)
        return counts

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "detllm_version": __version__,
            "artifact_type": "phase_diagram",
            "axes": self.axes,
            "dry_run": self.dry_run,
            "summary": self.summary(),
            "cells": [cell.to_dict() for cell in self.cells],
        }


def parse_axis_values(axis_items: list[str]) -> dict[str, list[Any]]:
    if not axis_items:
        raise ValueError("At least one phase axis is required")
    axes: dict[str, list[Any]] = {}
    for item in axis_items:
        if "=" not in item:
            raise ValueError("Phase axes must use name=value1,value2 syntax")
        name, raw_values = item.split("=", 1)
        name = name.strip()
        if name not in SUPPORTED_AXES:
            raise ValueError(f"Unsupported phase axis: {name}")
        raw_parts = [value.strip() for value in raw_values.split(",")]
        values = [_coerce_axis_value(name, value) for value in raw_parts if value]
        if not values:
            raise ValueError(f"Phase axis {name} must include at least one value")
        axes[name] = values
    return axes


def expand_grid(axes: dict[str, list[Any]], max_cells: int | None) -> list[PhaseCell]:
    ordered_names = sorted(axes)
    axis_values = [sorted(axes[name], key=lambda value: str(value)) for name in ordered_names]
    cells: list[PhaseCell] = []
    for index, values in enumerate(product(*axis_values)):
        cell_axes = dict(zip(ordered_names, values, strict=True))
        cell_id = "__".join(f"{name}-{_safe_value(cell_axes[name])}" for name in ordered_names)
        execution_status = "planned"
        if max_cells is not None and index >= max_cells:
            execution_status = "skipped"
        cells.append(
            PhaseCell(
                cell_id=cell_id,
                axes=cell_axes,
                execution_status=execution_status,
            )
        )
    return cells


def classify_cell(report: dict[str, Any], diagnosis: dict[str, Any]) -> str:
    causes = [cause.get("code") for cause in diagnosis.get("causes", [])]
    if report.get("status") == "PASS":
        if "score_margin_instability" in causes:
            return "fragile"
        return "stable"
    return "unstable"


def phase_from_cells(
    cells: list[PhaseCell],
    *,
    axes: dict[str, list[Any]],
    out_dir: str,
    dry_run: bool,
) -> PhaseDiagram:
    return PhaseDiagram(axes=axes, cells=cells, out_dir=out_dir, dry_run=dry_run)


def cell_artifact_path(out_dir: str, cell: PhaseCell) -> str:
    return os.path.join(out_dir, "cells", cell.cell_id)


def _coerce_axis_value(name: str, value: str) -> Any:
    if name in {"batch_size", "max_new_tokens"}:
        coerced = int(value)
        if coerced <= 0:
            raise ValueError(f"Phase axis {name} values must be positive integers")
        return coerced
    return value


def _safe_value(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "_")
