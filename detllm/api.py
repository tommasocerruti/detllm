"""Public Python API."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Sequence

from detllm.backends.base import BackendAdapter
from detllm.core.artifacts import dump_json
from detllm.core.capabilities import evaluate_capabilities
from detllm.core.deterministic import DeterministicContext
from detllm.core.env import capture_env
from detllm.diff.diff import aggregate_diffs, diff_traces
from detllm.report.report import Report
from detllm.report.render_text import render_report
from detllm.trace.io import write_trace


@dataclass(frozen=True)
class RunResult:
    status: str
    category: str
    out_dir: str


def run(
    *,
    backend: str,
    model: str,
    prompts: Sequence[str],
    tier: int = 1,
    mode: str = "best-effort",
    batch_size: int = 1,
    seed: int = 0,
    max_new_tokens: int = 32,
    device: str = "cpu",
    dtype: str = "float32",
    out_dir: str = "artifacts/run",
    backend_adapter: BackendAdapter | None = None,
) -> RunResult:
    from detllm.cli import main as cli_main
    if not prompts:
        raise ValueError("prompts must be non-empty")

    os.makedirs(out_dir, exist_ok=True)
    env_snapshot = capture_env()
    dump_json(os.path.join(out_dir, "env.json"), env_snapshot)

    args = _build_args(
        backend=backend,
        model=model,
        prompts=prompts,
        tier=tier,
        mode=mode,
        batch_size=batch_size,
        seed=seed,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        out_dir=out_dir,
    )

    with DeterministicContext(tier, mode, seed) as ctx:
        backend_impl = backend_adapter or cli_main._build_backend(args)
        decision = evaluate_capabilities(ctx.applied, backend_impl.capabilities(), tier, mode)
        if not decision.supported:
            cli_main._write_unsupported(out_dir, runs=1, decision=decision)
            dump_json(
                os.path.join(out_dir, "determinism_applied.json"),
                cli_main._wrap_artifact("determinism_applied", ctx.applied.to_dict()),
            )
            return RunResult(status="FAIL", category="UNSUPPORTED_REQUEST", out_dir=out_dir)

        trace_rows = cli_main._run_generation(
            backend_impl, list(prompts), args, capture_scores=ctx.applied.tier_effective >= 2
        )

    dump_json(
        os.path.join(out_dir, "determinism_applied.json"),
        cli_main._wrap_artifact("determinism_applied", ctx.applied.to_dict()),
    )
    run_config = cli_main._build_run_config(
        args,
        env_snapshot.get("device"),
        ctx.applied.tier_effective,
        cli_main._parse_vary_batch(None),
    )
    dump_json(os.path.join(out_dir, "run_config.json"), run_config)
    write_trace(os.path.join(out_dir, "trace.jsonl"), trace_rows)
    return RunResult(status="PASS", category="PASS", out_dir=out_dir)


def check(
    *,
    backend: str,
    model: str,
    prompts: Sequence[str],
    tier: int = 1,
    mode: str = "best-effort",
    runs: int = 3,
    batch_size: int = 1,
    vary_batch: Sequence[int] | None = None,
    seed: int = 0,
    max_new_tokens: int = 32,
    device: str = "cpu",
    dtype: str = "float32",
    out_dir: str = "artifacts/check",
    backend_adapter: BackendAdapter | None = None,
) -> Report:
    from detllm.cli import main as cli_main
    if not prompts:
        raise ValueError("prompts must be non-empty")

    os.makedirs(out_dir, exist_ok=True)
    env_snapshot = capture_env()
    dump_json(os.path.join(out_dir, "env.json"), env_snapshot)

    vary_batch_sizes = list(vary_batch or [])
    args = _build_args(
        backend=backend,
        model=model,
        prompts=prompts,
        tier=tier,
        mode=mode,
        batch_size=batch_size,
        seed=seed,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        out_dir=out_dir,
        runs=runs,
        vary_batch=vary_batch_sizes,
    )

    run_config = cli_main._build_run_config(
        args,
        env_snapshot.get("device"),
        tier_effective=tier,
        vary_batch=vary_batch_sizes,
    )
    dump_json(os.path.join(out_dir, "run_config.json"), run_config)

    traces: list[list[dict[str, Any]]] = []
    determinism_rows: list[dict[str, Any]] = []
    baseline_fingerprint = env_snapshot.get("fingerprint")
    for run_idx in range(runs):
        env_run = capture_env()
        env_path = os.path.join(out_dir, "envs", f"run_{run_idx}.json")
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        dump_json(env_path, env_run)
        if baseline_fingerprint and env_run.get("fingerprint") != baseline_fingerprint:
            cli_main._write_env_mismatch(out_dir, runs, run_idx, baseline_fingerprint, env_run)
            return Report(status="FAIL", category="ENV_MISMATCH", details={})
        with DeterministicContext(tier, mode, seed) as ctx:
            backend_impl = backend_adapter or cli_main._build_backend(args)
            decision = evaluate_capabilities(ctx.applied, backend_impl.capabilities(), tier, mode)
            if not decision.supported:
                cli_main._write_unsupported(out_dir, runs=runs, decision=decision)
                dump_json(
                    os.path.join(out_dir, "determinism_applied.json"),
                    cli_main._wrap_artifact("determinism_applied", ctx.applied.to_dict()),
                )
                return Report(status="FAIL", category="UNSUPPORTED_REQUEST", details={})

            trace_rows = cli_main._run_generation(
                backend_impl, list(prompts), args, capture_scores=ctx.applied.tier_effective >= 2
            )

        traces.append(trace_rows)
        determinism_rows.append(cli_main._wrap_artifact("determinism_applied", ctx.applied.to_dict()))
        trace_path = os.path.join(out_dir, "traces", f"run_{run_idx}.jsonl")
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        write_trace(trace_path, trace_rows)

    dump_json(os.path.join(out_dir, "determinism_applied.json"), determinism_rows[0])

    diffs = [diff_traces(traces[0], traces[i]) for i in range(1, len(traces))]
    result = aggregate_diffs(diffs)

    batch_result = None
    batch_traces: dict[int, list[dict[str, Any]]] = {}
    batch_diffs: list[tuple[int, Any]] = []
    if vary_batch_sizes:
        for batch_size_item in vary_batch_sizes:
            batch_args = cli_main._clone_args(args, batch_size=batch_size_item)
            with DeterministicContext(tier, mode, seed) as ctx:
                backend_impl = backend_adapter or cli_main._build_backend(batch_args)
                trace_rows = cli_main._run_generation(
                    backend_impl,
                    list(prompts),
                    batch_args,
                    capture_scores=ctx.applied.tier_effective >= 2,
                )
            batch_traces[batch_size_item] = trace_rows
            trace_path = os.path.join(out_dir, "traces", f"batch_{batch_size_item}.jsonl")
            write_trace(trace_path, trace_rows)

        batch_diffs = [
            (size, diff_traces(traces[0], batch_traces[size])) for size in vary_batch_sizes
        ]
        batch_result = aggregate_diffs([diff for _, diff in batch_diffs])

    report = Report(
        status=cli_main._report_status(result, batch_result),
        category=cli_main._report_category(result, batch_result),
        details={
            "runs": runs,
            "batch_sizes": vary_batch_sizes,
            "first_divergence": cli_main._report_divergence(result, batch_result),
            "batch_divergence": cli_main._batch_divergence_detail(batch_diffs, result),
        },
    )
    dump_json(os.path.join(out_dir, "report.json"), cli_main._wrap_artifact("report", report.to_dict()))
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(render_report(report))

    if cli_main._report_divergence(result, batch_result) is not None:
        diff_path = os.path.join(out_dir, "diffs", "first_divergence.json")
        dump_json(
            diff_path,
            cli_main._wrap_artifact("first_divergence", cli_main._report_divergence(result, batch_result)),
        )

    return report


def _build_args(**kwargs: Any) -> Any:
    class _Args:
        pass

    args = _Args()
    for key, value in kwargs.items():
        setattr(args, key, value)
    if not hasattr(args, "runs"):
        setattr(args, "runs", 1)
    if not hasattr(args, "vary_batch"):
        setattr(args, "vary_batch", [])
    return args
