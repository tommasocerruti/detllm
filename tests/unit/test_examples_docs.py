import json
import subprocess
import sys
from pathlib import Path

from detllm import analyze, recommend
from detllm.core.artifacts import load_json

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
PHASE_FIXTURE = EXAMPLES / "fixtures" / "phase_demo"


def test_example_prompt_files_are_valid_jsonl():
    prompt_files = [
        EXAMPLES / "prompts" / "basic.jsonl",
        EXAMPLES / "prompts" / "research_debugging.jsonl",
    ]

    for path in prompt_files:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows
        assert all(isinstance(row.get("prompt"), str) and row["prompt"] for row in rows)


def test_phase_demo_fixture_loads_and_supports_analysis_and_recommend(tmp_path):
    payload = load_json(str(PHASE_FIXTURE / "phase_diagram.json"))

    assert payload["artifact_type"] == "phase_diagram"
    assert payload["cells"]

    analysis = analyze(
        str(PHASE_FIXTURE),
        out_dir=str(tmp_path / "analysis"),
        confidence=0.95,
    )
    plan = recommend(
        str(PHASE_FIXTURE),
        out_dir=str(tmp_path / "recommendations"),
        budget_cells=4,
    )

    assert analysis.summary["executed"] > 0
    assert (tmp_path / "analysis" / "analysis.json").exists()
    assert plan.recommendations
    assert (tmp_path / "recommendations" / "experiment_plan.json").exists()


def test_example_fixture_cli_workflows(tmp_path):
    analysis_out = tmp_path / "analysis_cli"
    recommend_out = tmp_path / "recommend_cli"

    analyze_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "analyze",
            "--in",
            str(PHASE_FIXTURE),
            "--out",
            str(analysis_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    recommend_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "recommend",
            "--in",
            str(PHASE_FIXTURE),
            "--out",
            str(recommend_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert analyze_result.stderr == ""
    assert recommend_result.stderr == ""
    assert (analysis_out / "analysis.txt").exists()
    assert (recommend_out / "experiment_plan.txt").exists()


def test_docs_link_examples_and_cover_core_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    examples_doc = (ROOT / "docs" / "examples.md").read_text(encoding="utf-8")

    assert "docs/examples.md" in readme
    for command in ["check", "diagnose", "replay", "phase", "analyze", "recommend"]:
        assert f"detllm {command}" in examples_doc
