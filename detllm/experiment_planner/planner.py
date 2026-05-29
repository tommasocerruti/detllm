"""Adaptive experiment planner for reproducibility phase diagrams."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from detllm.analysis.analysis import analyze_phase_artifact
from detllm.core.artifacts import dump_json, load_json, validate_artifact
from detllm.version import __version__

SUPPORTED_STRATEGIES = {"auto", "coverage", "fragility", "boundary"}
ORDERED_AXES = {"batch_size", "max_new_tokens"}


@dataclass(frozen=True)
class RecommendedExperiment:
    rank: int
    recommendation_type: str
    score: float
    axes: dict[str, Any]
    reason: str
    source_cell_ids: list[str]
    suggested_overrides: dict[str, Any]
    command_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "recommendation_type": self.recommendation_type,
            "score": self.score,
            "axes": self.axes,
            "reason": self.reason,
            "source_cell_ids": self.source_cell_ids,
            "suggested_overrides": self.suggested_overrides,
            "command_hint": self.command_hint,
        }


@dataclass(frozen=True)
class ExperimentPlan:
    strategy: str
    budget_cells: int
    out_dir: str
    phase_dir: str | None
    analysis_summary: dict[str, Any]
    recommendations: list[RecommendedExperiment]

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "detllm_version": __version__,
            "artifact_type": "experiment_plan",
            "strategy": self.strategy,
            "budget_cells": self.budget_cells,
            "phase_dir": self.phase_dir,
            "analysis_summary": self.analysis_summary,
            "recommendations": [item.to_dict() for item in self.recommendations],
        }


@dataclass(frozen=True)
class _Candidate:
    recommendation_type: str
    score: float
    axes: dict[str, Any]
    reason: str
    source_cell_ids: list[str]
    suggested_overrides: dict[str, Any]


def recommend_phase_directory(
    in_dir: str,
    *,
    out_dir: str | None = None,
    budget_cells: int = 8,
    strategy: str = "auto",
    confidence: float = 0.95,
    validate_schema: bool = False,
) -> ExperimentPlan:
    phase_path = os.path.join(in_dir, "phase_diagram.json")
    if not os.path.exists(phase_path):
        raise FileNotFoundError(f"Missing phase_diagram.json in {in_dir}")

    phase_payload = load_json(phase_path)
    analysis_path = os.path.join(in_dir, "analysis", "analysis.json")
    analysis_payload = load_json(analysis_path) if os.path.exists(analysis_path) else None
    plan = plan_phase_artifact(
        phase_payload,
        analysis_payload=analysis_payload,
        phase_dir=in_dir,
        out_dir=out_dir or os.path.join(in_dir, "recommendations"),
        budget_cells=budget_cells,
        strategy=strategy,
        confidence=confidence,
    )
    _write_plan_artifacts(plan, validate_schema=validate_schema)
    return plan


def plan_phase_artifact(
    phase_payload: dict[str, Any],
    *,
    analysis_payload: dict[str, Any] | None = None,
    phase_dir: str | None = None,
    out_dir: str,
    budget_cells: int = 8,
    strategy: str = "auto",
    confidence: float = 0.95,
) -> ExperimentPlan:
    if strategy not in SUPPORTED_STRATEGIES:
        choices = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(f"Unsupported recommendation strategy {strategy}; expected {choices}")
    if budget_cells < 0:
        raise ValueError("budget_cells must be non-negative")

    analysis_summary = _analysis_summary(phase_payload, analysis_payload, confidence)
    cells = list(phase_payload.get("cells", []))
    candidates = _candidate_experiments(cells, strategy, analysis_payload)
    selected = _rank_candidates(candidates, budget_cells)
    recommendations = [
        RecommendedExperiment(
            rank=index + 1,
            recommendation_type=item.recommendation_type,
            score=round(item.score, 6),
            axes=item.axes,
            reason=item.reason,
            source_cell_ids=item.source_cell_ids,
            suggested_overrides=item.suggested_overrides,
            command_hint=_command_hint(item),
        )
        for index, item in enumerate(selected)
    ]
    return ExperimentPlan(
        strategy=strategy,
        budget_cells=budget_cells,
        out_dir=out_dir,
        phase_dir=phase_dir,
        analysis_summary=analysis_summary,
        recommendations=recommendations,
    )


def _write_plan_artifacts(plan: ExperimentPlan, *, validate_schema: bool = False) -> None:
    from detllm.experiment_planner.render import render_plan_text, write_plan_csv

    payload = plan.to_artifact()
    if validate_schema:
        validate_artifact(payload)
    dump_json(os.path.join(plan.out_dir, "experiment_plan.json"), payload)
    write_plan_csv(plan, os.path.join(plan.out_dir, "experiment_plan.csv"))
    with open(
        os.path.join(plan.out_dir, "experiment_plan.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write(render_plan_text(plan))


def _analysis_summary(
    phase_payload: dict[str, Any],
    analysis_payload: dict[str, Any] | None,
    confidence: float,
) -> dict[str, Any]:
    if analysis_payload is not None:
        return dict(analysis_payload.get("summary", {}))
    return analyze_phase_artifact(
        phase_payload,
        confidence=confidence,
        out_dir="",
    ).summary


def _candidate_experiments(
    cells: list[dict[str, Any]],
    strategy: str,
    analysis_payload: dict[str, Any] | None,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    if strategy in {"auto", "coverage"}:
        candidates.extend(_coverage_candidates(cells))
    if strategy in {"auto", "fragility"}:
        candidates.extend(_fragility_candidates(cells))
    if strategy == "auto":
        candidates.extend(_cluster_candidates(cells))
    if strategy in {"auto", "boundary"}:
        candidates.extend(_boundary_candidates(cells))
    if strategy == "auto":
        candidates.extend(_uncertainty_candidates(cells, analysis_payload))
    return candidates


def _coverage_candidates(cells: list[dict[str, Any]]) -> list[_Candidate]:
    candidates = []
    for cell in cells:
        if cell.get("execution_status") not in {"skipped", "planned"}:
            continue
        status = cell.get("execution_status")
        candidates.append(
            _Candidate(
                recommendation_type="complete_coverage",
                score=1000.0 if status == "skipped" else 950.0,
                axes=dict(cell.get("axes", {})),
                reason=f"Cell was {status}; running it removes missing grid coverage.",
                source_cell_ids=[cell.get("cell_id", "")],
                suggested_overrides={},
            )
        )
    return candidates


def _fragility_candidates(cells: list[dict[str, Any]]) -> list[_Candidate]:
    candidates = []
    for cell in cells:
        if cell.get("execution_status") != "executed":
            continue
        if cell.get("classification") != "fragile":
            continue
        margin = cell.get("min_topk_margin")
        margin_bonus = 0.0 if margin is None else max(0.0, 1.0 - float(margin))
        candidates.append(
            _Candidate(
                recommendation_type="rerun_fragile_cell",
                score=800.0 + margin_bonus,
                axes=dict(cell.get("axes", {})),
                reason=(
                    "Fragile score-margin behavior was observed; rerun with deeper "
                    "top-k capture."
                ),
                source_cell_ids=[cell.get("cell_id", "")],
                suggested_overrides={"capture_topk_scores": 10},
            )
        )
    return candidates


def _cluster_candidates(cells: list[dict[str, Any]]) -> list[_Candidate]:
    clusters: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        if cell.get("execution_status") != "executed":
            continue
        if cell.get("classification") != "unstable":
            continue
        for axis, value in cell.get("axes", {}).items():
            key = (str(axis), str(value))
            clusters.setdefault(key, []).append(cell)

    candidates = []
    for (axis, value), cluster_cells in sorted(clusters.items()):
        if len(cluster_cells) < 2:
            continue
        source_ids = sorted(cell.get("cell_id", "") for cell in cluster_cells)
        candidates.append(
            _Candidate(
                recommendation_type="focus_axis_value",
                score=700.0 + 10.0 * len(cluster_cells),
                axes={axis: value},
                reason=(
                    f"Unstable cells cluster at {axis}={value}; focus a sweep "
                    "around this value."
                ),
                source_cell_ids=source_ids,
                suggested_overrides={"focus_axis": axis, "focus_value": value},
            )
        )
    return candidates


def _boundary_candidates(cells: list[dict[str, Any]]) -> list[_Candidate]:
    candidates = []
    for axis in sorted(ORDERED_AXES):
        axis_cells = [
            cell
            for cell in cells
            if cell.get("execution_status") == "executed"
            and cell.get("classification") in {"stable", "fragile", "unstable"}
            and axis in cell.get("axes", {})
            and _is_number(cell["axes"][axis])
        ]
        grouped: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
        for cell in axis_cells:
            key = tuple(
                sorted(
                    (str(name), str(value))
                    for name, value in cell.get("axes", {}).items()
                    if name != axis
                )
            )
            grouped.setdefault(key, []).append(cell)
        for group_cells in grouped.values():
            ordered = sorted(group_cells, key=lambda cell: float(cell["axes"][axis]))
            for left, right in zip(ordered, ordered[1:], strict=False):
                left_class = left.get("classification")
                right_class = right.get("classification")
                if left_class == right_class:
                    continue
                source_ids = sorted([left.get("cell_id", ""), right.get("cell_id", "")])
                score = 600.0 + 10.0 * max(
                    _class_weight(left_class), _class_weight(right_class)
                )
                candidates.append(
                    _Candidate(
                        recommendation_type="boundary_probe",
                        score=score,
                        axes=_merge_axes(left, right),
                        reason=(
                            f"{axis} transitions from {left_class} to {right_class}; "
                            "probe the boundary."
                        ),
                        source_cell_ids=source_ids,
                        suggested_overrides={
                            "axis": axis,
                            "between": [left["axes"][axis], right["axes"][axis]],
                            "runs": 10,
                        },
                    )
                )
    return candidates


def _uncertainty_candidates(
    cells: list[dict[str, Any]],
    analysis_payload: dict[str, Any] | None,
) -> list[_Candidate]:
    if analysis_payload is None:
        return []
    interval = analysis_payload.get("intervals", {}).get("nonstable_rate", {})
    lower = interval.get("lower")
    upper = interval.get("upper")
    if lower is None or upper is None or float(upper) - float(lower) <= 0.5:
        return []

    candidates = []
    seen_classes = set()
    for cell in sorted(cells, key=lambda item: item.get("cell_id", "")):
        classification = cell.get("classification")
        if cell.get("execution_status") != "executed" or classification in seen_classes:
            continue
        if classification not in {"stable", "fragile", "unstable"}:
            continue
        seen_classes.add(classification)
        candidates.append(
            _Candidate(
                recommendation_type="reduce_uncertainty",
                score=400.0 + _class_weight(classification),
                axes=dict(cell.get("axes", {})),
                reason="Confidence interval is wide; rerun representative cells.",
                source_cell_ids=[cell.get("cell_id", "")],
                suggested_overrides={"runs": 10},
            )
        )
    return candidates


def _rank_candidates(
    candidates: list[_Candidate], budget_cells: int
) -> list[_Candidate]:
    deduped: dict[tuple[str, str, tuple[str, ...]], _Candidate] = {}
    for item in candidates:
        key = (
            item.recommendation_type,
            json.dumps(item.axes, sort_keys=True),
            tuple(item.source_cell_ids),
        )
        existing = deduped.get(key)
        if existing is None or item.score > existing.score:
            deduped[key] = item
    ranked = sorted(
        deduped.values(),
        key=lambda item: (
            -item.score,
            item.recommendation_type,
            json.dumps(item.axes, sort_keys=True),
            ",".join(item.source_cell_ids),
        ),
    )
    return ranked[:budget_cells]


def _command_hint(candidate: _Candidate) -> str:
    axis_args = " ".join(
        f"--axis {name}={value}" for name, value in sorted(candidate.axes.items())
    )
    override_args = " ".join(
        f"--{name.replace('_', '-')} {value}"
        for name, value in sorted(candidate.suggested_overrides.items())
        if name in {"capture_topk_scores", "runs"}
    )
    return " ".join(
        part for part in ["detllm phase", axis_args, override_args] if part
    )


def _merge_axes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    axes = dict(left.get("axes", {}))
    for name, value in right.get("axes", {}).items():
        if axes.get(name) != value:
            axes[name] = [axes.get(name), value]
    return axes


def _class_weight(classification: str | None) -> int:
    return {"stable": 0, "fragile": 1, "unstable": 2}.get(str(classification), 0)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
