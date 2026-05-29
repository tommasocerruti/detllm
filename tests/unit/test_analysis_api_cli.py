import json
import subprocess
import sys

import pytest

from detllm import analyze


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
                "cell_id": "fragile-a",
                "axes": {"batch_size": 2},
                "execution_status": "executed",
                "classification": "fragile",
                "report_status": "PASS",
                "report_category": "PASS",
                "first_divergence": None,
                "min_topk_margin": 0.001,
                "diagnosis_causes": ["score_margin_instability"],
                "artifact_path": "cells/fragile-a",
            },
        ],
    }
    (path / "phase_diagram.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_analyze_api_writes_artifacts(tmp_path):
    phase_dir = tmp_path / "phase"
    _write_phase_diagram(phase_dir)

    result = analyze(in_dir=str(phase_dir), confidence=0.95)

    assert result.risk_score == 25.0
    assert (phase_dir / "analysis" / "analysis.json").exists()
    assert (phase_dir / "analysis" / "analysis.csv").exists()
    assert (phase_dir / "analysis" / "analysis.txt").exists()


def test_analyze_cli_writes_artifacts(tmp_path):
    phase_dir = tmp_path / "phase"
    out_dir = tmp_path / "analysis"
    _write_phase_diagram(phase_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "analyze",
            "--in",
            str(phase_dir),
            "--out",
            str(out_dir),
            "--confidence",
            "0.95",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    payload = json.loads((out_dir / "analysis.json").read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "analysis"
    assert payload["summary"]["executed"] == 2


def test_analyze_missing_phase_diagram_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing phase_diagram.json"):
        analyze(in_dir=str(tmp_path / "missing"))
