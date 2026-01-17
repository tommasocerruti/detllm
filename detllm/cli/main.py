"""DetLLM command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

from detllm.backends.base import BackendAdapter
from detllm.backends.hf import HFBackend
from detllm.backends.vllm import VLLMBackend
from detllm.core.artifacts import (
    dump_json,
    load_json,
    load_schema,
    validate_artifact,
    validate_json,
)
from detllm.core.capabilities import evaluate_capabilities
from detllm.core.deterministic import DeterministicContext
from detllm.core.env import capture_env
from detllm.core.models import DeterminismAppliedRecord, EnvSnapshot, RunConfig, TokenTraceRow
from detllm.diff.diff import aggregate_diffs, diff_traces
from detllm.report.render_text import render_report
from detllm.report.report import Report
from detllm.trace.io import read_trace, write_trace
from detllm.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detllm",
        description=(
            "Deterministic-mode checks for LLM inference: measure run/batch variance, "
            "generate repro packs, and explain why outputs differ."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="Capture an environment snapshot")
    env_parser.add_argument(
        "--out",
        required=False,
        default="artifacts/env.json",
        help="Output path for env.json",
    )
    env_parser.add_argument(
        "--redact-env",
        action="store_true",
        help="Redact sensitive environment fields",
    )
    env_parser.add_argument(
        "--redact-env-var",
        action="append",
        default=[],
        help="Environment variable name to redact (repeatable)",
    )
    env_parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate output artifacts against schemas",
    )

    run_parser = subparsers.add_parser("run", help="Run a single inference and emit artifacts")
    run_parser.add_argument("--backend", required=False, default="hf", help="Backend adapter")
    run_parser.add_argument("--model", required=False, help="Model id or path")
    run_parser.add_argument("--prompt", required=False, help="Single prompt")
    run_parser.add_argument("--prompt-file", required=False, help="JSONL file of prompts")
    run_parser.add_argument("--tier", type=int, default=1, help="Determinism tier")
    run_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    run_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for determinism controls (defaults to 0 when omitted)",
    )
    run_parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens")
    run_parser.add_argument("--dtype", default="float32", help="Model dtype")
    run_parser.add_argument("--device", default="cpu", help="Device")
    run_parser.add_argument("--mode", choices=["strict", "best-effort"], default="best-effort")
    run_parser.add_argument(
        "--out",
        required=False,
        default="artifacts/run",
        help="Output directory for artifacts",
    )
    run_parser.add_argument(
        "--redact-env",
        action="store_true",
        help="Redact sensitive environment fields",
    )
    run_parser.add_argument(
        "--redact-env-var",
        action="append",
        default=[],
        help="Environment variable name to redact (repeatable)",
    )
    run_parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate output artifacts against schemas",
    )

    check_parser = subparsers.add_parser("check", help="Repeat runs and measure variance")
    check_parser.add_argument("--backend", required=False, default="hf", help="Backend adapter")
    check_parser.add_argument("--model", required=False, help="Model id or path")
    check_parser.add_argument("--prompt", required=False, help="Single prompt")
    check_parser.add_argument("--prompt-file", required=False, help="JSONL file of prompts")
    check_parser.add_argument("--tier", type=int, default=1, help="Determinism tier")
    check_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    check_parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    check_parser.add_argument(
        "--vary-batch",
        required=False,
        help="Comma-separated batch sizes to measure batch variance",
    )
    check_parser.add_argument("--seed", type=int, default=0, help="Seed for determinism controls")
    check_parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens")
    check_parser.add_argument("--dtype", default="float32", help="Model dtype")
    check_parser.add_argument("--device", default="cpu", help="Device")
    check_parser.add_argument("--mode", choices=["strict", "best-effort"], default="best-effort")
    check_parser.add_argument(
        "--out",
        required=False,
        default="artifacts/check",
        help="Output directory for artifacts",
    )
    check_parser.add_argument(
        "--redact-env",
        action="store_true",
        help="Redact sensitive environment fields",
    )
    check_parser.add_argument(
        "--redact-env-var",
        action="append",
        default=[],
        help="Environment variable name to redact (repeatable)",
    )
    check_parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate output artifacts against schemas",
    )

    diff_parser = subparsers.add_parser("diff", help="Diff traces and emit a report")
    diff_parser.add_argument("--left", required=False, help="Left trace file (jsonl)")
    diff_parser.add_argument("--right", required=False, help="Right trace file (jsonl)")
    diff_parser.add_argument(
        "--out",
        required=False,
        default="artifacts/diff",
        help="Output directory for diff artifacts",
    )
    diff_parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate output artifacts against schemas",
    )
    report_parser = subparsers.add_parser("report", help="Render report artifacts")
    report_parser.add_argument("--in", dest="report_in", required=False, help="Input report.json")
    report_parser.add_argument(
        "--out",
        required=False,
        default="artifacts/report.txt",
        help="Output text report path",
    )
    report_parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate report.json against schema",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "env":
        snapshot = capture_env(**_redact_kwargs(args))
        env_payload = _coerce_env(snapshot)
        if args.validate_schema:
            validate_artifact(env_payload)
        dump_json(args.out, env_payload)
        return 0

    if args.command == "run":
        if not args.model:
            parser.error("--model is required for run")

        prompts = _load_prompts(args)
        if not prompts:
            parser.error("Prompt input is required via --prompt or --prompt-file")

        os.makedirs(args.out, exist_ok=True)
        env_snapshot = capture_env(**_redact_kwargs(args))
        env_payload = _coerce_env(env_snapshot)
        if args.validate_schema:
            validate_artifact(env_payload)
        dump_json(os.path.join(args.out, "env.json"), env_payload)

        with DeterministicContext(args.tier, args.mode, args.seed) as ctx:
            backend = _build_backend(args)
            decision = evaluate_capabilities(ctx.applied, backend.capabilities(), args.tier, args.mode)
            if not decision.supported:
                _write_unsupported(args.out, args.runs if hasattr(args, "runs") else 1, decision)
                dump_json(
                    os.path.join(args.out, "determinism_applied.json"),
                    _coerce_determinism(ctx.applied.to_dict()),
                )
                return 2
            trace_rows = _run_generation(
                backend, prompts, args, capture_scores=ctx.applied.tier_effective >= 2
            )

        determinism_payload = _coerce_determinism(ctx.applied.to_dict())
        if args.validate_schema:
            validate_artifact(determinism_payload)
        dump_json(os.path.join(args.out, "determinism_applied.json"), determinism_payload)
        run_config = _build_run_config(
            args,
            env_snapshot.get("device"),
            ctx.applied.tier_effective,
            _parse_vary_batch(getattr(args, "vary_batch", None)),
        )
        run_config = _coerce_run_config(run_config)
        if args.validate_schema:
            validate_artifact(run_config)
        dump_json(os.path.join(args.out, "run_config.json"), run_config)
        write_trace(
            os.path.join(args.out, "trace.jsonl"),
            _coerce_trace_rows(trace_rows),
            validate_rows=args.validate_schema,
        )
        return 0

    if args.command == "check":
        if not args.model:
            parser.error("--model is required for check")

        prompts = _load_prompts(args)
        if not prompts:
            parser.error("Prompt input is required via --prompt or --prompt-file")

        os.makedirs(args.out, exist_ok=True)
        env_snapshot = capture_env(**_redact_kwargs(args))
        env_payload = _coerce_env(env_snapshot)
        if args.validate_schema:
            validate_artifact(env_payload)
        dump_json(os.path.join(args.out, "env.json"), env_payload)

        vary_batch_sizes = _parse_vary_batch(args.vary_batch)
        run_config = _build_run_config(
            args,
            env_snapshot.get("device"),
            tier_effective=args.tier,
            vary_batch=vary_batch_sizes,
        )
        run_config = _coerce_run_config(run_config)
        if args.validate_schema:
            validate_artifact(run_config)
        dump_json(os.path.join(args.out, "run_config.json"), run_config)

        traces: list[list[dict[str, Any]]] = []
        determinism_rows: list[dict[str, Any]] = []
        baseline_fingerprint = env_snapshot.get("fingerprint")
        # TODO: Consider reusing the backend per run for speed; per-run reloads isolate state.
        for run_idx in range(args.runs):
            env_run = capture_env(**_redact_kwargs(args))
            env_payload = _coerce_env(env_run)
            env_path = os.path.join(args.out, "envs", f"run_{run_idx}.json")
            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            dump_json(env_path, env_payload)
            if baseline_fingerprint and env_payload.get("fingerprint") != baseline_fingerprint:
                _write_env_mismatch(args.out, args.runs, run_idx, baseline_fingerprint, env_payload)
                return 0
            with DeterministicContext(args.tier, args.mode, args.seed) as ctx:
                backend = _build_backend(args)
                decision = evaluate_capabilities(
                    ctx.applied, backend.capabilities(), args.tier, args.mode
                )
                if not decision.supported:
                    _write_unsupported(args.out, args.runs, decision)
                    dump_json(
                        os.path.join(args.out, "determinism_applied.json"),
                        _coerce_determinism(ctx.applied.to_dict()),
                    )
                    return 2
                trace_rows = _run_generation(
                    backend, prompts, args, capture_scores=ctx.applied.tier_effective >= 2
                )

            traces.append(trace_rows)
            determinism_rows.append(_coerce_determinism(ctx.applied.to_dict()))
            trace_path = os.path.join(args.out, "traces", f"run_{run_idx}.jsonl")
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            write_trace(
                trace_path,
                _coerce_trace_rows(trace_rows),
                validate_rows=args.validate_schema,
            )

        # Determinism controls are expected to be stable across runs; record first run only.
        if args.validate_schema:
            validate_artifact(determinism_rows[0])
        dump_json(os.path.join(args.out, "determinism_applied.json"), determinism_rows[0])

        diffs = [
            diff_traces(traces[0], traces[i]) for i in range(1, len(traces))
        ]
        result = aggregate_diffs(diffs)

        batch_result = None
        batch_traces: dict[int, list[dict[str, Any]]] = {}
        batch_diffs: list[tuple[int, Any]] = []
        if vary_batch_sizes:
            for batch_size in vary_batch_sizes:
                batch_args = _clone_args(args, batch_size=batch_size)
                with DeterministicContext(args.tier, args.mode, args.seed) as ctx:
                    backend = _build_backend(batch_args)
                    trace_rows = _run_generation(
                        backend,
                        prompts,
                        batch_args,
                        capture_scores=ctx.applied.tier_effective >= 2,
                    )
                batch_traces[batch_size] = trace_rows
                trace_path = os.path.join(args.out, "traces", f"batch_{batch_size}.jsonl")
                write_trace(trace_path, _coerce_trace_rows(trace_rows))

            # Compare against the baseline (fixed batch) trace only, not pairwise.
            batch_diffs = [
                (size, diff_traces(traces[0], batch_traces[size]))
                for size in vary_batch_sizes
            ]
            batch_result = aggregate_diffs([diff for _, diff in batch_diffs])

        report = Report(
            status=_report_status(result, batch_result),
            category=_report_category(result, batch_result),
            details={
                "runs": args.runs,
                "batch_sizes": vary_batch_sizes,
                "first_divergence": _report_divergence(result, batch_result),
                "batch_divergence": _batch_divergence_detail(batch_diffs, result),
            },
        )
        report_payload = _wrap_artifact("report", report.to_dict())
        if args.validate_schema:
            validate_artifact(report_payload)
        dump_json(os.path.join(args.out, "report.json"), report_payload)
        report_text = render_report(report)
        with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as handle:
            handle.write(report_text)

        if _report_divergence(result, batch_result) is not None:
            diff_path = os.path.join(args.out, "diffs", "first_divergence.json")
            dump_json(
                diff_path,
                _wrap_artifact("first_divergence", _report_divergence(result, batch_result)),
            )

        return 0

    if args.command == "diff":
        if not args.left or not args.right:
            parser.error("--left and --right are required for diff")

        os.makedirs(args.out, exist_ok=True)
        left_trace = read_trace(args.left)
        right_trace = read_trace(args.right)
        result = diff_traces(left_trace, right_trace)

        report = Report(
            status=result.status,
            category=result.category,
            details={
                "first_divergence": result.first_divergence,
            },
        )
        report_payload = _wrap_artifact("report", report.to_dict())
        if args.validate_schema:
            validate_artifact(report_payload)
        dump_json(os.path.join(args.out, "report.json"), report_payload)
        report_text = render_report(report)
        with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as handle:
            handle.write(report_text)

        if result.first_divergence is not None:
            diff_path = os.path.join(args.out, "diffs", "first_divergence.json")
            dump_json(
                diff_path,
                _wrap_artifact("first_divergence", result.first_divergence),
            )
        return 0

    if args.command == "report":
        if not args.report_in:
            parser.error("--in is required for report")

        payload = load_json(args.report_in)
        if args.validate_schema:
            validate_json(payload, load_schema("report"))
        report = Report(
            status=payload["status"],
            category=payload["category"],
            details=payload.get("details", {}),
        )
        report_text = render_report(report)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report_text)
        return 0

    return 0


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [args.prompt]

    if args.prompt_file:
        prompts: list[str] = []
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if isinstance(data, str):
                    prompts.append(data)
                elif isinstance(data, dict):
                    prompt = data.get("prompt") or data.get("text")
                    if prompt is None:
                        raise ValueError("Prompt file entries must include 'prompt' or 'text'")
                    prompts.append(prompt)
                else:
                    raise ValueError("Prompt file entries must be JSON objects or strings")
        return prompts

    return []


def _build_backend(args: argparse.Namespace) -> BackendAdapter:
    if args.backend == "hf":
        return HFBackend(args.model, device=args.device, dtype=args.dtype)
    if args.backend == "vllm":
        return VLLMBackend(args.model)
    raise ValueError(f"Unsupported backend: {args.backend}")


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _hash_token_ids(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_generation(
    backend: BackendAdapter,
    prompts: list[str],
    args: argparse.Namespace,
    capture_scores: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results = backend.generate(
            batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            capture_scores=capture_scores,
        )
        for item in results:
            rows.append(
                {
                    "prompt_id": _hash_prompt(item["prompt"]),
                    "input_token_ids": item["input_ids"],
                    # TODO: Add a privacy mode to store only token hashes/redacted ids.
                    "input_token_ids_hash": _hash_token_ids(item["input_ids"]),
                    "generated_token_ids": item["output_ids"],
                    "scores": item.get("scores"),
                }
            )
    return rows


def _build_run_config(
    args: argparse.Namespace,
    device_snapshot: dict[str, Any] | None,
    tier_effective: int,
    vary_batch: list[int],
) -> dict[str, Any]:
    data = {
        "backend": args.backend,
        "tier_requested": args.tier,
        "tier_effective": tier_effective,
        "mode": args.mode,
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        "batch_size": args.batch_size,
        "vary_batch": vary_batch,
        "tokenizer": {
            "id": args.model,
            # TODO: Capture tokenizer revision + class metadata.
        },
        "generation_context": {
            "device_snapshot": device_snapshot,
        },
    }
    return _wrap_artifact("run_config", data)


def _wrap_artifact(artifact_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "detllm_version": __version__,
        "artifact_type": artifact_type,
        **payload,
    }


def _redact_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "redact": getattr(args, "redact_env", False),
        "redact_env_vars": getattr(args, "redact_env_var", []),
    }


def _coerce_env(payload: dict[str, Any]) -> dict[str, Any]:
    return EnvSnapshot.from_dict(payload).to_dict()


def _coerce_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    return RunConfig.from_dict(payload).to_dict()


def _coerce_determinism(payload: dict[str, Any]) -> dict[str, Any]:
    if "schema_version" not in payload:
        payload = _wrap_artifact("determinism_applied", payload)
    return DeterminismAppliedRecord.from_dict(payload).to_dict()


def _coerce_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [TokenTraceRow.from_dict(row).to_dict() for row in rows]


def _write_unsupported(out_dir: str, runs: int, decision) -> None:
    report = Report(
        status="FAIL",
        category="UNSUPPORTED_REQUEST",
        details={
            "runs": runs,
            "capability_failures": decision.capability_failures,
            "notes": getattr(decision, "notes", []),
        },
    )
    dump_json(
        os.path.join(out_dir, "report.json"),
        _wrap_artifact("report", report.to_dict()),
    )
    report_text = render_report(report)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(report_text)


def _write_env_mismatch(
    out_dir: str,
    runs: int,
    run_index: int,
    baseline_fingerprint: str,
    current_env: dict[str, Any],
) -> None:
    report = Report(
        status="FAIL",
        category="ENV_MISMATCH",
        details={
            "runs": runs,
            "run_index": run_index,
            "baseline_fingerprint": baseline_fingerprint,
            "current_fingerprint": current_env.get("fingerprint"),
        },
    )
    dump_json(
        os.path.join(out_dir, "report.json"),
        _wrap_artifact("report", report.to_dict()),
    )
    report_text = render_report(report)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(report_text)


def _parse_vary_batch(value: str | None) -> list[int]:
    if not value:
        return []
    sizes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            raise ValueError("Batch sizes must be positive integers")
        sizes.append(size)
    return sizes


def _clone_args(args: argparse.Namespace, batch_size: int) -> argparse.Namespace:
    data = vars(args).copy()
    data["batch_size"] = batch_size
    return argparse.Namespace(**data)


def _report_status(result, batch_result) -> str:
    if result.status != "PASS":
        return result.status
    if batch_result and batch_result.status != "PASS":
        return "FAIL"
    return "PASS"


def _report_category(result, batch_result) -> str:
    if result.status != "PASS":
        return result.category
    if batch_result and batch_result.status != "PASS":
        return "BATCH_VARIANCE"
    return "PASS"


def _report_divergence(result, batch_result):
    if result.status != "PASS":
        return result.first_divergence
    if batch_result and batch_result.status != "PASS":
        return batch_result.first_divergence
    return None


def _batch_divergence_detail(batch_diffs, run_result):
    if run_result.status != "PASS":
        return None
    for batch_size, diff in batch_diffs:
        if diff.status != "PASS":
            return {
                "batch_size": batch_size,
                "first_divergence": diff.first_divergence,
            }
    return None


if __name__ == "__main__":
    raise SystemExit(main())
