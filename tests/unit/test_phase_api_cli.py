import json
import subprocess
import sys

from detllm import phase
from detllm.backends.base import BackendCapabilities


class StableBackend:
    def __init__(self):
        self.calls = 0

    def capabilities(self):
        return BackendCapabilities(
            supports_tier1_fixed_batch=True,
            supports_scores=True,
            supports_torch_deterministic=True,
        )

    def generate(self, prompts, **kwargs):
        self.calls += 1
        return [
            {
                "prompt": prompt,
                "input_ids": [1],
                "output_ids": [1, 2],
                "scores": [-0.1],
                "topk_token_ids": [[2, 3]],
                "topk_scores": [[-0.1, -0.5]],
                "tokenizer_id": "fake",
            }
            for prompt in prompts
        ]


def test_phase_api_dry_run_writes_artifacts_without_backend_calls(tmp_path):
    backend = StableBackend()

    diagram = phase(
        backend="hf",
        model="fake",
        prompts=["hello"],
        axes={"batch_size": [1, 2]},
        runs=2,
        out_dir=str(tmp_path / "phase"),
        dry_run=True,
        backend_adapter=backend,
    )

    assert backend.calls == 0
    assert diagram.summary()["planned"] == 2
    assert (tmp_path / "phase" / "phase_diagram.json").exists()
    assert (tmp_path / "phase" / "phase_diagram.csv").exists()
    assert (tmp_path / "phase" / "phase_diagram.txt").exists()


def test_phase_api_executes_cells_and_writes_cell_artifacts(tmp_path):
    backend = StableBackend()

    diagram = phase(
        backend="hf",
        model="fake",
        prompts=["hello"],
        axes={"batch_size": [1]},
        runs=2,
        out_dir=str(tmp_path / "phase"),
        backend_adapter=backend,
    )

    assert diagram.summary()["stable"] == 1
    assert diagram.cells[0].artifact_path.endswith("cells/batch_size-1")
    assert (tmp_path / "phase" / "cells" / "batch_size-1" / "report.json").exists()
    assert (tmp_path / "phase" / "cells" / "batch_size-1" / "diagnosis.json").exists()


def test_phase_cli_dry_run_writes_phase_artifacts(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    out_dir = tmp_path / "phase"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "phase",
            "--backend",
            "hf",
            "--model",
            "fake",
            "--prompt-file",
            str(prompt_file),
            "--axis",
            "batch_size=1,2",
            "--dry-run",
            "--out",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    payload = json.loads((out_dir / "phase_diagram.json").read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "phase_diagram"
    assert payload["summary"]["planned"] == 2
