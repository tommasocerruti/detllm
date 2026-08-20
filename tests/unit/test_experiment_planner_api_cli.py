import json
import subprocess
import sys

from detllm import recommend


def _write_phase_diagram(path):
    path.mkdir()
    payload = {
        "schema_version": "1.0",
        "detllm_version": "0.1.1",
        "artifact_type": "phase_diagram",
        "axes": {"batch_size": [1, 2]},
        "dry_run": False,
        "summary": {},
        "cells": [
            {
                "cell_id": "stable-a",
                "axes": {"batch_size": 1},
                "execution_status": "executed",
                "classification": "stable",
                "report_status": "PASS",
                "report_category": "PASS",
                "first_divergence": None,
                "min_topk_margin": None,
                "diagnosis_causes": [],
                "artifact_path": "cells/stable-a",
            },
            {
                "cell_id": "skipped-a",
                "axes": {"batch_size": 2},
                "execution_status": "skipped",
                "classification": None,
                "report_status": None,
                "report_category": None,
                "first_divergence": None,
                "min_topk_margin": None,
                "diagnosis_causes": [],
                "artifact_path": None,
            },
        ],
    }
    (path / "phase_diagram.json").write_text(json.dumps(payload), encoding="utf-8")


def test_recommend_api_writes_artifacts(tmp_path):
    phase_dir = tmp_path / "phase"
    _write_phase_diagram(phase_dir)

    plan = recommend(str(phase_dir), budget_cells=4)

    assert plan.recommendations[0].recommendation_type == "complete_coverage"
    assert (phase_dir / "recommendations" / "experiment_plan.json").exists()
    assert (phase_dir / "recommendations" / "experiment_plan.csv").exists()
    assert (phase_dir / "recommendations" / "experiment_plan.txt").exists()


def test_recommend_cli_writes_artifacts(tmp_path):
    phase_dir = tmp_path / "phase"
    out_dir = tmp_path / "recommendations"
    _write_phase_diagram(phase_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "recommend",
            "--in",
            str(phase_dir),
            "--out",
            str(out_dir),
            "--budget-cells",
            "4",
            "--strategy",
            "auto",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    payload = json.loads((out_dir / "experiment_plan.json").read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "experiment_plan"
    assert payload["recommendations"][0]["recommendation_type"] == "complete_coverage"
