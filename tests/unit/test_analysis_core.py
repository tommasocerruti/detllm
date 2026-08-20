import json

import pytest

from detllm.analysis.analysis import (
    analyze_phase_artifact,
    risk_score,
    wilson_interval,
)
from detllm.core.artifacts import load_schema


def _phase_payload(cells):
    return {
        "schema_version": "1.0",
        "detllm_version": "0.1.1",
        "artifact_type": "phase_diagram",
        "axes": {"batch_size": [1, 2]},
        "dry_run": False,
        "summary": {},
        "cells": cells,
    }


def _cell(cell_id, classification, execution_status="executed", axes=None):
    return {
        "cell_id": cell_id,
        "axes": axes or {"batch_size": 1},
        "execution_status": execution_status,
        "classification": classification,
        "report_status": "PASS" if classification != "unstable" else "FAIL",
        "report_category": "PASS" if classification != "unstable" else "RUN_VARIANCE_FIXED_BATCH",
        "first_divergence": None,
        "min_topk_margin": None,
        "diagnosis_causes": [],
        "artifact_path": f"cells/{cell_id}",
    }


def test_wilson_interval_zero_all_and_mixed_cases():
    zero = wilson_interval(successes=0, total=10, confidence=0.95)
    all_fail = wilson_interval(successes=10, total=10, confidence=0.95)
    mixed = wilson_interval(successes=5, total=10, confidence=0.95)

    assert zero["lower"] == 0.0
    assert 0.0 < zero["upper"] < 0.4
    assert 0.6 < all_fail["lower"] < 1.0
    assert all_fail["upper"] == 1.0
    assert mixed["lower"] < 0.5 < mixed["upper"]


def test_wilson_interval_rejects_unsupported_confidence():
    with pytest.raises(ValueError, match="Unsupported confidence"):
        wilson_interval(successes=1, total=2, confidence=0.8)


def test_risk_score_weights_fragile_cells_halfway():
    assert risk_score(stable=3, fragile=0, unstable=0) == 0.0
    assert risk_score(stable=0, fragile=0, unstable=3) == 100.0
    assert risk_score(stable=1, fragile=2, unstable=1) == 50.0
    assert risk_score(stable=0, fragile=0, unstable=0) is None


def test_analyze_phase_artifact_computes_summary_intervals_and_recommendations():
    payload = _phase_payload(
        [
            _cell("stable-a", "stable", axes={"batch_size": 1}),
            _cell("fragile-a", "fragile", axes={"batch_size": 2}),
            _cell("unstable-a", "unstable", axes={"batch_size": 4}),
            _cell("unstable-b", "unstable", axes={"batch_size": 4}),
            _cell("skipped-a", None, execution_status="skipped", axes={"batch_size": 8}),
        ]
    )

    result = analyze_phase_artifact(payload, confidence=0.95, out_dir="analysis")

    assert result.summary["executed"] == 4
    assert result.summary["fragile"] == 1
    assert result.summary["unstable"] == 2
    assert result.risk_score == 62.5
    assert result.intervals["unstable_rate"]["count"] == 2
    codes = [item["code"] for item in result.recommendations]
    assert "rerun_fragile_cells" in codes
    assert "complete_skipped_grid" in codes
    assert "focus_axis_value" in codes
    json.dumps(result.to_artifact())


def test_analyze_phase_artifact_rule_of_three_and_schema_available():
    payload = _phase_payload([_cell(f"stable-{idx}", "stable") for idx in range(5)])

    result = analyze_phase_artifact(payload, confidence=0.95, out_dir="analysis")

    assert result.summary["unstable"] == 0
    assert "3/N" in result.summary["rule_of_three"]
    assert load_schema("analysis")["properties"]["artifact_type"]["const"] == "analysis"
