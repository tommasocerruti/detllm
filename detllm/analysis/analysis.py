"""Statistical reproducibility analysis."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from detllm.core.artifacts import dump_json, load_json, validate_artifact
from detllm.version import __version__

SUPPORTED_CONFIDENCE_Z = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


@dataclass(frozen=True)
class AnalysisResult:
    summary: dict[str, Any]
    intervals: dict[str, dict[str, Any]]
    risk_score: float | None
    recommendations: list[dict[str, Any]]
    cells: list[dict[str, Any]]
    out_dir: str

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "detllm_version": __version__,
            "artifact_type": "analysis",
            "summary": self.summary,
            "intervals": self.intervals,
            "risk_score": self.risk_score,
            "recommendations": self.recommendations,
            "cells": self.cells,
        }


def wilson_interval(
    *, successes: int, total: int, confidence: float = 0.95
) -> dict[str, Any]:
    confidence = _normalize_confidence(confidence)
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    if total == 0:
        return {
            "count": successes,
            "total": total,
            "rate": None,
            "lower": None,
            "upper": None,
            "confidence": confidence,
        }

    z = SUPPORTED_CONFIDENCE_Z[confidence]
    rate = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (rate + z2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt((rate * (1 - rate) + z2 / (4 * total)) / total)
        / denominator
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return {
        "count": successes,
        "total": total,
        "rate": _round(rate),
        "lower": _round(lower),
        "upper": _round(upper),
        "confidence": confidence,
    }


def risk_score(*, stable: int, fragile: int, unstable: int) -> float | None:
    executed = stable + fragile + unstable
    if executed == 0:
        return None
    return _round(100 * (unstable + 0.5 * fragile) / executed)


def analyze_phase_directory(
    in_dir: str,
    *,
    confidence: float = 0.95,
    out_dir: str | None = None,
    validate_schema: bool = False,
) -> AnalysisResult:
    phase_path = os.path.join(in_dir, "phase_diagram.json")
    if not os.path.exists(phase_path):
        raise FileNotFoundError(f"Missing phase_diagram.json in {in_dir}")

    phase_payload = load_json(phase_path)
    result = analyze_phase_artifact(
        phase_payload,
        confidence=confidence,
        out_dir=out_dir or os.path.join(in_dir, "analysis"),
    )
    _write_analysis_artifacts(result, validate_schema=validate_schema)
    return result


def analyze_phase_artifact(
    phase_payload: dict[str, Any],
    *,
    confidence: float = 0.95,
    out_dir: str,
) -> AnalysisResult:
    confidence = _normalize_confidence(confidence)
    cells = [_derive_cell(cell) for cell in phase_payload.get("cells", [])]
    executed_cells = [cell for cell in cells if cell["execution_status"] == "executed"]
    stable = sum(1 for cell in executed_cells if cell["classification"] == "stable")
    fragile = sum(1 for cell in executed_cells if cell["classification"] == "fragile")
    unstable = sum(1 for cell in executed_cells if cell["classification"] == "unstable")
    skipped = sum(1 for cell in cells if cell["execution_status"] == "skipped")
    planned = sum(1 for cell in cells if cell["execution_status"] == "planned")
    executed = len(executed_cells)

    intervals = {
        "unstable_rate": wilson_interval(
            successes=unstable, total=executed, confidence=confidence
        ),
        "nonstable_rate": wilson_interval(
            successes=fragile + unstable, total=executed, confidence=confidence
        ),
    }
    summary = {
        "total_cells": len(cells),
        "executed": executed,
        "stable": stable,
        "fragile": fragile,
        "unstable": unstable,
        "skipped": skipped,
        "planned": planned,
        "unstable_rate": _rate(unstable, executed),
        "nonstable_rate": _rate(fragile + unstable, executed),
        "rule_of_three": _rule_of_three(unstable, executed, confidence),
    }
    score = risk_score(stable=stable, fragile=fragile, unstable=unstable)
    return AnalysisResult(
        summary=summary,
        intervals=intervals,
        risk_score=score,
        recommendations=_recommendations(
            cells=cells,
            intervals=intervals,
            executed=executed,
            fragile=fragile,
            skipped=skipped,
        ),
        cells=cells,
        out_dir=out_dir,
    )


def _write_analysis_artifacts(
    result: AnalysisResult, *, validate_schema: bool = False
) -> None:
    from detllm.analysis.render import render_analysis_text, write_analysis_csv

    payload = result.to_artifact()
    if validate_schema:
        validate_artifact(payload)
    dump_json(os.path.join(result.out_dir, "analysis.json"), payload)
    write_analysis_csv(result, os.path.join(result.out_dir, "analysis.csv"))
    with open(
        os.path.join(result.out_dir, "analysis.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write(render_analysis_text(result))


def _derive_cell(cell: dict[str, Any]) -> dict[str, Any]:
    classification = cell.get("classification")
    risk_weight = None
    if cell.get("execution_status") == "executed":
        risk_weight = {"stable": 0.0, "fragile": 0.5, "unstable": 1.0}.get(
            classification
        )
    return {
        "cell_id": cell.get("cell_id", ""),
        "axes": cell.get("axes", {}),
        "execution_status": cell.get("execution_status", "planned"),
        "classification": classification,
        "risk_weight": risk_weight,
        "report_status": cell.get("report_status"),
        "report_category": cell.get("report_category"),
        "first_divergence": cell.get("first_divergence"),
        "min_topk_margin": cell.get("min_topk_margin"),
        "diagnosis_causes": cell.get("diagnosis_causes", []),
        "artifact_path": cell.get("artifact_path"),
    }


def _recommendations(
    *,
    cells: list[dict[str, Any]],
    intervals: dict[str, dict[str, Any]],
    executed: int,
    fragile: int,
    skipped: int,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    if _interval_width(intervals["nonstable_rate"]) > 0.5:
        recommendations.append(
            {
                "code": "increase_sample_size",
                "summary": (
                    "Confidence interval is wide; increase --runs or expand "
                    "--max-cells before drawing strong conclusions."
                ),
            }
        )
    if fragile:
        recommendations.append(
            {
                "code": "rerun_fragile_cells",
                "summary": (
                    "Fragile cells were observed; rerun them with higher "
                    "--capture-topk-scores to inspect score margins."
                ),
                "cell_count": fragile,
            }
        )
    if skipped:
        recommendations.append(
            {
                "code": "complete_skipped_grid",
                "summary": (
                    "Skipped cells remain; complete the planned grid before "
                    "claiming full coverage."
                ),
                "cell_count": skipped,
            }
        )
    recommendations.extend(_axis_cluster_recommendations(cells))
    if executed == 0:
        recommendations.append(
            {
                "code": "execute_phase_grid",
                "summary": (
                    "No executed cells were available; run the phase diagram "
                    "before statistical analysis."
                ),
            }
        )
    return recommendations


def _axis_cluster_recommendations(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for cell in cells:
        if cell["execution_status"] != "executed" or cell["classification"] != "unstable":
            continue
        for name, value in cell["axes"].items():
            key = (str(name), str(value))
            counts[key] = counts.get(key, 0) + 1

    recommendations = []
    for (axis, value), count in sorted(counts.items()):
        if count < 2:
            continue
        recommendations.append(
            {
                "code": "focus_axis_value",
                "summary": (
                    f"Unstable cells cluster at {axis}={value}; run a focused "
                    "sweep around that value."
                ),
                "axis": axis,
                "value": value,
                "unstable_cells": count,
            }
        )
    return recommendations


def _interval_width(interval: dict[str, Any]) -> float:
    lower = interval.get("lower")
    upper = interval.get("upper")
    if lower is None or upper is None:
        return 1.0
    return float(upper) - float(lower)


def _rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return _round(count / total)


def _rule_of_three(unstable: int, executed: int, confidence: float) -> str | None:
    if unstable != 0 or executed == 0:
        return None
    level = int(round(confidence * 100))
    return (
        f"No unstable cells observed; with N executed cells, the {level}% upper "
        "bound is approximately 3/N."
    )


def _normalize_confidence(confidence: float) -> float:
    for supported in SUPPORTED_CONFIDENCE_Z:
        if math.isclose(confidence, supported, rel_tol=0.0, abs_tol=1e-9):
            return supported
    supported_values = ", ".join(str(value) for value in sorted(SUPPORTED_CONFIDENCE_Z))
    raise ValueError(f"Unsupported confidence {confidence}; expected one of {supported_values}")


def _round(value: float) -> float:
    return round(value, 6)
