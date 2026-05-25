"""Offline diagnosis for detLLM check artifacts."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any

from detllm.core.artifacts import dump_json, load_json, validate_artifact
from detllm.flight_recorder.render_text import render_diagnosis
from detllm.trace.io import read_trace
from detllm.version import __version__


@dataclass(frozen=True)
class Diagnosis:
    status: str
    category: str
    summary: str
    causes: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    probe_plan: list[dict[str, Any]]
    out_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "category": self.category,
            "summary": self.summary,
            "causes": self.causes,
            "evidence": self.evidence,
            "probe_plan": self.probe_plan,
        }


def diagnose_directory(
    in_dir: str,
    out_dir: str | None = None,
    *,
    include_token_text: bool = False,
    validate_schema: bool = False,
) -> Diagnosis:
    """Analyze an existing detLLM check directory and write diagnosis artifacts."""

    target_dir = out_dir or in_dir
    report = load_json(os.path.join(in_dir, "report.json"))
    run_config = load_json(os.path.join(in_dir, "run_config.json"))
    determinism = _load_optional_json(os.path.join(in_dir, "determinism_applied.json"))
    traces = _load_traces(in_dir)

    evidence = _collect_evidence(report, run_config, determinism, traces)
    causes = _rank_causes(report, run_config, determinism, traces, evidence)
    probe_plan = _build_probe_plan(causes)
    summary = _summary(report, causes)

    diagnosis = Diagnosis(
        status=report.get("status", "UNKNOWN"),
        category=report.get("category", "UNKNOWN"),
        summary=summary,
        causes=causes,
        evidence=evidence,
        probe_plan=probe_plan,
        out_dir=target_dir,
    )
    payload = _wrap_diagnosis(diagnosis.to_dict())
    if validate_schema:
        validate_artifact(payload)
    dump_json(os.path.join(target_dir, "diagnosis.json"), payload)
    with open(os.path.join(target_dir, "diagnosis.txt"), "w", encoding="utf-8") as handle:
        handle.write(render_diagnosis(diagnosis, include_token_text=include_token_text))
    return diagnosis


def _load_optional_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    return load_json(path)


def _load_traces(in_dir: str) -> dict[str, list[dict[str, Any]]]:
    trace_dir = os.path.join(in_dir, "traces")
    traces: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        traces[name] = read_trace(path)
    single_trace = os.path.join(in_dir, "trace.jsonl")
    if os.path.exists(single_trace):
        traces["trace"] = read_trace(single_trace)
    return traces


def _collect_evidence(
    report: dict[str, Any],
    run_config: dict[str, Any],
    determinism: dict[str, Any],
    traces: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    details = report.get("details", {})
    evidence: list[dict[str, Any]] = [
        {
            "code": "report_category",
            "message": f"Report category is {report.get('category', 'UNKNOWN')}.",
            "data": {"status": report.get("status"), "category": report.get("category")},
        }
    ]
    if details.get("batch_divergence"):
        evidence.append(
            {
                "code": "batch_divergence",
                "message": "A vary-batch trace diverged from the fixed-batch baseline.",
                "data": details["batch_divergence"],
            }
        )
    if details.get("first_divergence"):
        evidence.append(
            {
                "code": "first_divergence",
                "message": "The report includes a first divergence location.",
                "data": details["first_divergence"],
            }
        )
    if determinism.get("capability_failures") or determinism.get("downgrades"):
        evidence.append(
            {
                "code": "capability_limits",
                "message": "Requested guarantees were limited by backend capabilities.",
                "data": {
                    "capability_failures": determinism.get("capability_failures", []),
                    "downgrades": determinism.get("downgrades", []),
                },
            }
        )
    if run_config.get("backend") == "vllm":
        evidence.append(
            {
                "code": "vllm_tier0",
                "message": "The vLLM adapter is measurement-only in detLLM v1.",
                "data": {"backend": "vllm"},
            }
        )
    margin = _smallest_topk_margin(traces)
    if margin is not None:
        evidence.append(
            {
                "code": "small_score_margin",
                "message": "A top-k score margin is small enough to make token choice fragile.",
                "data": margin,
            }
        )
    return evidence


def _rank_causes(
    report: dict[str, Any],
    run_config: dict[str, Any],
    determinism: dict[str, Any],
    traces: dict[str, list[dict[str, Any]]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    causes: list[dict[str, Any]] = []
    category = report.get("category")
    details = report.get("details", {})
    evidence_codes = {item["code"] for item in evidence}

    if category == "ENV_MISMATCH":
        causes.append(
            _cause(
                "environment_drift",
                "Environment drift",
                "high",
                "A run environment fingerprint changed during the check.",
                "Compare envs/run_*.json and rerun in a stable environment.",
            )
        )
    if category == "TOKENIZATION_MISMATCH":
        causes.append(
            _cause(
                "tokenization_drift",
                "Tokenization drift",
                "high",
                "Input token hashes or tokenizer identifiers differ across traces.",
                "Pin tokenizer revision and compare tokenizer metadata.",
            )
        )
    if category == "BATCH_VARIANCE" or details.get("batch_divergence"):
        causes.append(
            _cause(
                "batch_shape_drift",
                "Batch-shape numeric drift",
                "high",
                "Fixed-batch runs match, but at least one alternate batch size diverges.",
                "Run the batch-shape replay probe to isolate the sensitive batch size.",
            )
        )
    if "small_score_margin" in evidence_codes:
        causes.append(
            _cause(
                "score_margin_instability",
                "Score margin instability",
                "medium",
                "A top-1/top-2 score margin is very small before or at divergence.",
                "Rerun with top-k score capture and inspect nearby logits.",
            )
        )
    if determinism.get("capability_failures") or determinism.get("downgrades"):
        causes.append(
            _cause(
                "unsupported_guarantee",
                "Unsupported guarantee",
                "high",
                "The backend could not enforce the requested determinism tier.",
                "Use strict mode or a backend with the required capability.",
            )
        )
    if run_config.get("backend") == "vllm":
        causes.append(
            _cause(
                "measurement_only_backend",
                "Measurement-only backend",
                "medium",
                "The vLLM adapter captures traces but does not claim Tier 1/2 guarantees.",
                "Use HF for Tier 1/2 checks or treat vLLM output as measurement only.",
            )
        )
    if not causes and report.get("status") == "PASS":
        causes.append(
            _cause(
                "no_divergence_detected",
                "No divergence detected",
                "high",
                "The available traces match under the requested checks.",
                "No replay probe is required unless you want broader coverage.",
            )
        )
    elif not causes:
        causes.append(
            _cause(
                "unclassified_divergence",
                "Unclassified divergence",
                "low",
                "The repro pack failed but does not match a known Flight Recorder heuristic.",
                "Inspect first_divergence and rerun with Tier 2/top-k capture.",
            )
        )
    return causes


def _cause(
    code: str,
    title: str,
    confidence: str,
    explanation: str,
    next_step: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "title": title,
        "confidence": confidence,
        "explanation": explanation,
        "next_step": next_step,
    }


def _build_probe_plan(causes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    cause_codes = [cause["code"] for cause in causes]
    mapping = {
        "batch_shape_drift": ("batch-shape", "Reproduce divergence across batch sizes."),
        "score_margin_instability": ("score-margins", "Capture top-k score margins."),
        "unsupported_guarantee": ("tier2", "Retry with score-capable Tier 2 where supported."),
        "tokenization_drift": ("isolate-prompts", "Rerun divergent prompt in isolation."),
        "unclassified_divergence": ("isolate-prompts", "Minimize the failing prompt context."),
    }
    for code in cause_codes:
        probe = mapping.get(code)
        if probe is None:
            continue
        if any(item["probe"] == probe[0] for item in probes):
            continue
        probes.append({"probe": probe[0], "reason": probe[1]})
    return probes


def _smallest_topk_margin(
    traces: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for trace_name, rows in traces.items():
        for row_index, row in enumerate(rows):
            topk_scores = row.get("topk_scores")
            if not topk_scores:
                continue
            for token_index, scores in enumerate(topk_scores):
                if len(scores) < 2:
                    continue
                margin = abs(float(scores[0]) - float(scores[1]))
                if best is None or margin < best["margin"]:
                    best = {
                        "trace": trace_name,
                        "row_index": row_index,
                        "token_index": token_index,
                        "margin": margin,
                    }
    if best is not None and best["margin"] <= 1e-3:
        return best
    return None


def _summary(report: dict[str, Any], causes: list[dict[str, Any]]) -> str:
    category = report.get("category", "UNKNOWN")
    if not causes:
        return f"{category}: no likely cause identified."
    return f"{category}: most likely cause is {causes[0]['title']}."


def _wrap_diagnosis(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "detllm_version": __version__,
        "artifact_type": "diagnosis",
        **payload,
    }
