import json

import pytest

from detllm.core.artifacts import load_schema
from detllm.experiment_planner.planner import (
    plan_phase_artifact,
    recommend_phase_directory,
)


def _phase_payload(cells):
    return {
        "schema_version": "1.0",
        "detllm_version": "0.1.1",
        "artifact_type": "phase_diagram",
        "axes": {"batch_size": [1, 2, 4, 8], "dtype": ["float32", "bfloat16"]},
        "dry_run": False,
        "summary": {},
        "cells": cells,
    }


def _cell(cell_id, classification, axes, execution_status="executed"):
    return {
        "cell_id": cell_id,
        "axes": axes,
        "execution_status": execution_status,
        "classification": classification,
        "report_status": "PASS" if classification != "unstable" else "FAIL",
        "report_category": "PASS" if classification != "unstable" else "RUN_VARIANCE_FIXED_BATCH",
        "first_divergence": None,
        "min_topk_margin": 0.001 if classification == "fragile" else None,
        "diagnosis_causes": ["score_margin_instability"] if classification == "fragile" else [],
        "artifact_path": f"cells/{cell_id}",
    }


def test_auto_strategy_prioritizes_skipped_cells_and_applies_budget():
    payload = _phase_payload(
        [
            _cell("stable-1", "stable", {"batch_size": 1, "dtype": "float32"}),
            _cell("fragile-2", "fragile", {"batch_size": 2, "dtype": "float32"}),
            _cell("unstable-4", "unstable", {"batch_size": 4, "dtype": "bfloat16"}),
            _cell("skipped-8", None, {"batch_size": 8, "dtype": "bfloat16"}, "skipped"),
        ]
    )

    plan = plan_phase_artifact(payload, budget_cells=2, strategy="auto", out_dir="rec")

    assert len(plan.recommendations) == 2
    assert plan.recommendations[0].recommendation_type == "complete_coverage"
    assert plan.recommendations[0].axes == {"batch_size": 8, "dtype": "bfloat16"}
    assert plan.recommendations[0].score > plan.recommendations[1].score
    json.dumps(plan.to_artifact())


def test_fragility_strategy_recommends_higher_topk_capture():
    payload = _phase_payload(
        [
            _cell("stable-1", "stable", {"batch_size": 1}),
            _cell("fragile-2", "fragile", {"batch_size": 2}),
        ]
    )

    plan = plan_phase_artifact(payload, budget_cells=8, strategy="fragility", out_dir="rec")

    assert [item.recommendation_type for item in plan.recommendations] == [
        "rerun_fragile_cell"
    ]
    assert plan.recommendations[0].suggested_overrides["capture_topk_scores"] == 10
    assert "score-margin" in plan.recommendations[0].reason


def test_auto_strategy_emits_unstable_axis_cluster_recommendation():
    payload = _phase_payload(
        [
            _cell("unstable-a", "unstable", {"batch_size": 4, "dtype": "bfloat16"}),
            _cell("unstable-b", "unstable", {"batch_size": 8, "dtype": "bfloat16"}),
            _cell("stable-a", "stable", {"batch_size": 1, "dtype": "float32"}),
        ]
    )

    plan = plan_phase_artifact(payload, budget_cells=8, strategy="auto", out_dir="rec")
    cluster = [
        item
        for item in plan.recommendations
        if item.recommendation_type == "focus_axis_value"
    ]

    assert cluster
    assert cluster[0].axes == {"dtype": "bfloat16"}
    assert cluster[0].source_cell_ids == ["unstable-a", "unstable-b"]


def test_boundary_strategy_detects_numeric_class_transitions():
    payload = _phase_payload(
        [
            _cell("stable-1", "stable", {"batch_size": 1, "dtype": "float32"}),
            _cell("fragile-2", "fragile", {"batch_size": 2, "dtype": "float32"}),
            _cell("unstable-4", "unstable", {"batch_size": 4, "dtype": "float32"}),
        ]
    )

    plan = plan_phase_artifact(payload, budget_cells=8, strategy="boundary", out_dir="rec")

    assert [item.recommendation_type for item in plan.recommendations] == [
        "boundary_probe",
        "boundary_probe",
    ]
    assert plan.recommendations[0].suggested_overrides["axis"] == "batch_size"
    assert plan.recommendations[0].source_cell_ids == ["fragile-2", "unstable-4"]


def test_unsupported_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unsupported recommendation strategy"):
        plan_phase_artifact(_phase_payload([]), strategy="random", out_dir="rec")


def test_directory_loader_computes_analysis_without_writing_it(tmp_path):
    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    payload = _phase_payload(
        [_cell("stable-1", "stable", {"batch_size": 1, "dtype": "float32"})]
    )
    (phase_dir / "phase_diagram.json").write_text(json.dumps(payload), encoding="utf-8")

    plan = recommend_phase_directory(str(phase_dir), budget_cells=4)

    assert plan.analysis_summary["executed"] == 1
    assert not (phase_dir / "analysis" / "analysis.json").exists()
    assert load_schema("experiment_plan")["properties"]["artifact_type"]["const"] == (
        "experiment_plan"
    )


def test_missing_phase_diagram_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing phase_diagram.json"):
        recommend_phase_directory(str(tmp_path / "missing"))
