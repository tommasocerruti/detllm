import json
import subprocess
import sys

from detllm import diagnose, replay
from detllm.backends.base import BackendCapabilities


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _base_report(category="BATCH_VARIANCE"):
    if category == "PASS":
        details = {
            "runs": 2,
            "batch_sizes": [],
            "first_divergence": None,
            "batch_divergence": None,
        }
    else:
        details = {
            "runs": 2,
            "batch_sizes": [1, 2],
            "first_divergence": {"index": 0, "token_index": 1},
            "batch_divergence": {
                "batch_size": 2,
                "first_divergence": {"index": 0, "token_index": 1},
            },
        }
    return {
        "schema_version": "1.0",
        "detllm_version": "0.1.1",
        "artifact_type": "report",
        "status": "FAIL" if category != "PASS" else "PASS",
        "category": category,
        "details": details,
    }


def _base_run_config():
    return {
        "schema_version": "1.0",
        "detllm_version": "0.1.1",
        "artifact_type": "run_config",
        "backend": "hf",
        "tier_requested": 1,
        "tier_effective": 1,
        "mode": "best-effort",
        "model": "fake",
        "dtype": "float32",
        "device": "cpu",
        "decoding": {"max_new_tokens": 4, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
        "batch_size": 1,
        "vary_batch": [2],
        "tokenizer": {"id": "fake"},
        "generation_context": {},
    }


def _base_determinism():
    return {
        "schema_version": "1.0",
        "detllm_version": "0.1.1",
        "artifact_type": "determinism_applied",
        "tier_requested": 1,
        "tier_effective": 1,
        "mode": "best-effort",
        "seed": 0,
        "seed_controls": {},
        "torch_controls": {},
        "env_controls": {},
        "downgrades": [],
        "warnings": [],
        "capability_failures": [],
    }


def _write_minimal_check_dir(root):
    _write_json(root / "report.json", _base_report())
    _write_json(root / "run_config.json", _base_run_config())
    _write_json(root / "determinism_applied.json", _base_determinism())
    _write_trace(
        root / "traces" / "run_0.jsonl",
        [{"prompt_id": "p0", "generated_token_ids": [10, 20]}],
    )
    _write_trace(
        root / "traces" / "batch_2.jsonl",
        [{"prompt_id": "p0", "generated_token_ids": [10, 21]}],
    )


class ReplayBackend:
    def __init__(self):
        self.calls = []

    def capabilities(self):
        return BackendCapabilities(
            supports_tier1_fixed_batch=True,
            supports_scores=True,
            supports_torch_deterministic=True,
        )

    def generate(self, prompts, **kwargs):
        self.calls.append((list(prompts), dict(kwargs)))
        return [
            {
                "prompt": prompt,
                "input_ids": [1],
                "output_ids": [1, 2],
                "scores": [-0.1],
                "topk_token_ids": [[2, 3]],
                "topk_scores": [[-0.1, -0.2]],
            }
            for prompt in prompts
        ]


def test_public_diagnose_api_writes_artifact(tmp_path):
    check_dir = tmp_path / "check"
    _write_minimal_check_dir(check_dir)

    diagnosis = diagnose(str(check_dir))

    assert diagnosis.causes[0]["code"] == "batch_shape_drift"
    assert (check_dir / "diagnosis.json").exists()


def test_replay_auto_runs_only_diagnosis_backed_probe_with_prompt_text(tmp_path):
    check_dir = tmp_path / "check"
    _write_minimal_check_dir(check_dir)
    _write_trace(
        check_dir / "traces" / "run_0.jsonl",
        [{"prompt_id": "p0", "prompt_text": "hello", "generated_token_ids": [1, 2]}],
    )
    backend = ReplayBackend()

    result = replay(str(check_dir), probe="auto", backend_adapter=backend)

    assert result.status == "PASS"
    assert result.executed_probes == ["batch-shape"]
    assert [call[1]["capture_topk_scores"] for call in backend.calls] == [0, 0]
    assert (check_dir / "replay.json").exists()


def test_replay_skips_when_prompt_text_is_unavailable(tmp_path):
    check_dir = tmp_path / "check"
    _write_minimal_check_dir(check_dir)

    result = replay(str(check_dir), probe="batch-shape", backend_adapter=ReplayBackend())

    assert result.status == "SKIPPED"
    assert result.skipped_probes[0]["probe"] == "batch-shape"
    assert "prompt_text" in result.skipped_probes[0]["reason"]


def test_cli_diagnose_and_replay_write_artifacts(tmp_path):
    check_dir = tmp_path / "check"
    _write_minimal_check_dir(check_dir)
    _write_json(check_dir / "report.json", _base_report(category="PASS"))
    _write_json(check_dir / "run_config.json", _base_run_config())
    _write_trace(
        check_dir / "traces" / "run_0.jsonl",
        [{"prompt_id": "p0", "prompt_text": "hello", "generated_token_ids": [1, 2]}],
    )

    diagnose_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "diagnose",
            "--in",
            str(check_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert diagnose_result.stderr == ""
    assert (check_dir / "diagnosis.json").exists()

    replay_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detllm.cli.main",
            "--quiet",
            "replay",
            "--in",
            str(check_dir),
            "--probe",
            "auto",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert replay_result.stderr == ""
    payload = json.loads((check_dir / "replay.json").read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "replay"
