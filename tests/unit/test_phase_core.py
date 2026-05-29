import json

import pytest

from detllm.core.artifacts import load_schema
from detllm.phase_diagram.phase import (
    classify_cell,
    expand_grid,
    parse_axis_values,
    phase_from_cells,
)


def test_parse_axis_values_accepts_supported_axes():
    axes = parse_axis_values(
        [
            "batch_size=1,2",
            "dtype=float32,bfloat16",
            "max_new_tokens=16,32",
        ]
    )

    assert axes == {
        "batch_size": [1, 2],
        "dtype": ["float32", "bfloat16"],
        "max_new_tokens": [16, 32],
    }


def test_parse_axis_values_ignores_empty_values_before_numeric_coercion():
    axes = parse_axis_values(["batch_size=1,"])

    assert axes == {"batch_size": [1]}


def test_parse_axis_values_rejects_unknown_axis():
    with pytest.raises(ValueError, match="Unsupported phase axis"):
        parse_axis_values(["device=cpu,cuda"])


def test_parse_axis_values_rejects_empty_axis_list():
    with pytest.raises(ValueError, match="At least one phase axis"):
        parse_axis_values([])


def test_expand_grid_is_deterministic_and_marks_skipped_cells():
    cells = expand_grid(
        {
            "dtype": ["float32", "bfloat16"],
            "batch_size": [1, 2],
            "max_new_tokens": [8],
        },
        max_cells=2,
    )

    assert [cell.cell_id for cell in cells] == [
        "batch_size-1__dtype-bfloat16__max_new_tokens-8",
        "batch_size-1__dtype-float32__max_new_tokens-8",
        "batch_size-2__dtype-bfloat16__max_new_tokens-8",
        "batch_size-2__dtype-float32__max_new_tokens-8",
    ]
    assert [cell.execution_status for cell in cells] == [
        "planned",
        "planned",
        "skipped",
        "skipped",
    ]


def test_classify_cell_stable_fragile_unstable():
    stable = classify_cell(
        report={"status": "PASS", "category": "PASS", "details": {}},
        diagnosis={"causes": []},
    )
    fragile = classify_cell(
        report={"status": "PASS", "category": "PASS", "details": {}},
        diagnosis={"causes": [{"code": "score_margin_instability"}]},
    )
    unstable = classify_cell(
        report={
            "status": "FAIL",
            "category": "RUN_VARIANCE_FIXED_BATCH",
            "details": {"first_divergence": {"token_index": 3}},
        },
        diagnosis={"causes": [{"code": "batch_shape_drift"}]},
    )

    assert stable == "stable"
    assert fragile == "fragile"
    assert unstable == "unstable"


def test_phase_from_cells_counts_classes_and_schema_is_available(tmp_path):
    cells = expand_grid({"batch_size": [1, 2]}, max_cells=None)
    cells[0].classification = "stable"
    cells[0].execution_status = "executed"
    cells[1].classification = "unstable"
    cells[1].execution_status = "executed"

    diagram = phase_from_cells(
        cells,
        axes={"batch_size": [1, 2]},
        out_dir=str(tmp_path),
        dry_run=False,
    )

    payload = diagram.to_artifact()
    assert payload["artifact_type"] == "phase_diagram"
    assert payload["summary"]["stable"] == 1
    assert payload["summary"]["unstable"] == 1
    assert load_schema("phase_diagram")["properties"]["artifact_type"]["const"] == "phase_diagram"
    json.dumps(payload)
