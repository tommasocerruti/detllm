"""Targeted replay probes for detLLM check artifacts."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

from detllm.backends.base import BackendAdapter
from detllm.core.artifacts import dump_json, load_json, validate_artifact
from detllm.flight_recorder.diagnose import diagnose_directory
from detllm.trace.io import read_trace, write_trace
from detllm.version import __version__


@dataclass(frozen=True)
class ReplayResult:
    status: str
    out_dir: str
    executed_probes: list[str]
    skipped_probes: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "executed_probes": self.executed_probes,
            "skipped_probes": self.skipped_probes,
        }


def replay_directory(
    in_dir: str,
    *,
    probe: str = "auto",
    out_dir: str | None = None,
    include_token_text: bool = False,
    capture_topk_scores: int = 5,
    validate_schema: bool = False,
    backend_adapter: BackendAdapter | None = None,
) -> ReplayResult:
    target_dir = out_dir or in_dir
    run_config = load_json(os.path.join(in_dir, "run_config.json"))
    diagnosis = diagnose_directory(in_dir, out_dir=target_dir)
    probes = _resolve_probes(probe, diagnosis.probe_plan)
    prompts = _load_prompt_texts(in_dir)

    executed: list[str] = []
    skipped: list[dict[str, str]] = []
    for probe_name in probes:
        if not prompts:
            skipped.append(
                {
                    "probe": probe_name,
                    "reason": "Replay requires trace rows with opt-in prompt_text.",
                }
            )
            continue
        if probe_name == "batch-shape":
            _run_batch_shape_probe(
                in_dir,
                target_dir,
                run_config,
                prompts,
                include_token_text=include_token_text,
                backend_adapter=backend_adapter,
            )
            executed.append(probe_name)
        elif probe_name == "isolate-prompts":
            _run_generation_probe(
                target_dir,
                "isolate-prompts",
                run_config,
                prompts[:1],
                batch_size=1,
                capture_scores=False,
                capture_topk_scores=0,
                include_token_text=include_token_text,
                backend_adapter=backend_adapter,
            )
            executed.append(probe_name)
        elif probe_name in {"score-margins", "tier2"}:
            _run_generation_probe(
                target_dir,
                probe_name,
                run_config,
                prompts,
                batch_size=int(run_config.get("batch_size", 1)),
                capture_scores=True,
                capture_topk_scores=capture_topk_scores,
                include_token_text=include_token_text,
                backend_adapter=backend_adapter,
            )
            executed.append(probe_name)
        else:
            skipped.append({"probe": probe_name, "reason": "Unknown replay probe."})

    status = "PASS" if executed else "SKIPPED"
    result = ReplayResult(
        status=status,
        out_dir=target_dir,
        executed_probes=executed,
        skipped_probes=skipped,
    )
    payload = _wrap_replay(result.to_dict())
    if validate_schema:
        validate_artifact(payload)
    dump_json(os.path.join(target_dir, "replay.json"), payload)
    return result


def _resolve_probes(probe: str, probe_plan: list[dict[str, Any]]) -> list[str]:
    if probe == "auto":
        return [item["probe"] for item in probe_plan]
    return [probe]


def _load_prompt_texts(in_dir: str) -> list[str]:
    trace_path = os.path.join(in_dir, "traces", "run_0.jsonl")
    if not os.path.exists(trace_path):
        trace_path = os.path.join(in_dir, "trace.jsonl")
    if not os.path.exists(trace_path):
        return []
    prompts: list[str] = []
    for row in read_trace(trace_path):
        prompt_text = row.get("prompt_text")
        if prompt_text is None:
            return []
        prompts.append(prompt_text)
    return prompts


def _run_batch_shape_probe(
    in_dir: str,
    out_dir: str,
    run_config: dict[str, Any],
    prompts: list[str],
    *,
    include_token_text: bool,
    backend_adapter: BackendAdapter | None,
) -> None:
    batch_sizes = [int(run_config.get("batch_size", 1))]
    batch_sizes.extend(int(size) for size in run_config.get("vary_batch", []))
    for batch_size in dict.fromkeys(batch_sizes):
        _run_generation_probe(
            out_dir,
            f"batch-shape-{batch_size}",
            run_config,
            prompts,
            batch_size=batch_size,
            capture_scores=False,
            capture_topk_scores=0,
            include_token_text=include_token_text,
            backend_adapter=backend_adapter,
        )


def _run_generation_probe(
    out_dir: str,
    probe_name: str,
    run_config: dict[str, Any],
    prompts: list[str],
    *,
    batch_size: int,
    capture_scores: bool,
    capture_topk_scores: int,
    include_token_text: bool,
    backend_adapter: BackendAdapter | None,
) -> None:
    from detllm.cli import main as cli_main

    args = _args_from_run_config(run_config, batch_size, include_token_text)
    backend = backend_adapter or cli_main._build_backend(args)
    rows = cli_main._run_generation(
        backend,
        prompts,
        args,
        capture_scores=capture_scores,
        capture_topk_scores=capture_topk_scores,
    )
    trace_path = os.path.join(out_dir, "replays", f"{probe_name}.jsonl")
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    write_trace(trace_path, cli_main._coerce_trace_rows(rows))


def _args_from_run_config(
    run_config: dict[str, Any],
    batch_size: int,
    include_token_text: bool,
) -> argparse.Namespace:
    decoding = run_config.get("decoding", {})
    return argparse.Namespace(
        backend=run_config.get("backend", "hf"),
        model=run_config.get("model"),
        dtype=run_config.get("dtype", "float32"),
        device=run_config.get("device", "cpu"),
        batch_size=batch_size,
        max_new_tokens=int(decoding.get("max_new_tokens", 32)),
        temperature=float(decoding.get("temperature", 0.0)),
        top_p=float(decoding.get("top_p", 1.0)),
        top_k=int(decoding.get("top_k", 0)),
        include_token_text=include_token_text,
    )


def _wrap_replay(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "detllm_version": __version__,
        "artifact_type": "replay",
        **payload,
    }
