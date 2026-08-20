import json

from detllm.core.artifacts import load_schema
from detllm.flight_recorder.diagnose import diagnose_directory


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
    return {
        "schema_version": "1.0",
        "detllm_version": "0.1.1",
        "artifact_type": "report",
        "status": "FAIL",
        "category": category,
        "details": {
            "runs": 2,
            "batch_sizes": [1, 2],
            "first_divergence": {"index": 0, "token_index": 1},
            "batch_divergence": {
                "batch_size": 2,
                "first_divergence": {"index": 0, "token_index": 1},
            },
        },
    }


def _base_run_config(backend="hf"):
    return {
        "schema_version": "1.0",
        "detllm_version": "0.1.1",
        "artifact_type": "run_config",
        "backend": backend,
        "tier_requested": 1,
        "tier_effective": 1,
        "mode": "best-effort",
        "model": "fake",
        "dtype": "float32",
        "device": "cpu",
        "decoding": {"max_new_tokens": 4},
        "batch_size": 1,
        "vary_batch": [2],
        "tokenizer": {"id": "fake"},
        "generation_context": {},
    }


def _base_determinism(capability_failures=None, downgrades=None):
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
        "downgrades": downgrades or [],
        "warnings": [],
        "capability_failures": capability_failures or [],
    }


def _write_minimal_check_dir(root, *, report=None, run_config=None, determinism=None):
    _write_json(root / "report.json", report or _base_report())
    _write_json(root / "run_config.json", run_config or _base_run_config())
    _write_json(root / "determinism_applied.json", determinism or _base_determinism())
    _write_trace(
        root / "traces" / "run_0.jsonl",
        [{"prompt_id": "p0", "generated_token_ids": [10, 20]}],
    )
    _write_trace(
        root / "traces" / "run_1.jsonl",
        [{"prompt_id": "p0", "generated_token_ids": [10, 20]}],
    )
    _write_trace(
        root / "traces" / "batch_2.jsonl",
        [{"prompt_id": "p0", "generated_token_ids": [10, 21]}],
    )


def test_diagnose_directory_ranks_batch_shape_drift_and_writes_artifacts(tmp_path):
    check_dir = tmp_path / "check"
    _write_minimal_check_dir(check_dir)

    diagnosis = diagnose_directory(str(check_dir))

    assert diagnosis.status == "FAIL"
    assert diagnosis.category == "BATCH_VARIANCE"
    assert diagnosis.causes[0]["code"] == "batch_shape_drift"
    assert diagnosis.causes[0]["confidence"] == "high"
    assert diagnosis.probe_plan[0]["probe"] == "batch-shape"
    assert (check_dir / "diagnosis.json").exists()
    assert "Likely causes" in (check_dir / "diagnosis.txt").read_text(encoding="utf-8")


def test_diagnose_directory_detects_unsupported_guarantee_and_vllm(tmp_path):
    check_dir = tmp_path / "check"
    _write_minimal_check_dir(
        check_dir,
        report=_base_report(category="UNSUPPORTED_REQUEST"),
        run_config=_base_run_config(backend="vllm"),
        determinism=_base_determinism(
            capability_failures=[{"requirement": "tier1_fixed_batch"}],
            downgrades=[{"from": 1, "to": 0, "reason": "capability limits"}],
        ),
    )

    diagnosis = diagnose_directory(str(check_dir))

    codes = [cause["code"] for cause in diagnosis.causes]
    assert "unsupported_guarantee" in codes
    assert "measurement_only_backend" in codes
    assert any(item["probe"] == "tier2" for item in diagnosis.probe_plan)


def test_diagnose_directory_detects_score_margin_instability(tmp_path):
    check_dir = tmp_path / "check"
    _write_minimal_check_dir(check_dir, report=_base_report(category="RUN_VARIANCE_FIXED_BATCH"))
    _write_trace(
        check_dir / "traces" / "run_0.jsonl",
        [
            {
                "prompt_id": "p0",
                "generated_token_ids": [10, 20],
                "topk_token_ids": [[10, 11], [20, 21]],
                "topk_scores": [[-0.1, -1.1], [-0.2000, -0.2003]],
            }
        ],
    )

    diagnosis = diagnose_directory(str(check_dir))

    assert any(cause["code"] == "score_margin_instability" for cause in diagnosis.causes)
    assert any(item["probe"] == "score-margins" for item in diagnosis.probe_plan)


def test_flight_recorder_schemas_are_available():
    assert load_schema("diagnosis")["properties"]["artifact_type"]["const"] == "diagnosis"
    assert load_schema("replay")["properties"]["artifact_type"]["const"] == "replay"
