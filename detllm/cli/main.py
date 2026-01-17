"""DetLLM command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

from detllm.backends.hf import HFBackend
from detllm.core.artifacts import dump_json
from detllm.core.deterministic import DeterministicContext
from detllm.core.env import capture_env
from detllm.trace.io import write_trace
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

    check_parser = subparsers.add_parser("check", help="Repeat runs and measure variance")
    check_parser.add_argument("--backend", required=False, default="hf", help="Backend adapter")
    check_parser.add_argument("--model", required=False, help="Model id or path")
    check_parser.add_argument("--prompt", required=False, help="Single prompt")
    check_parser.add_argument("--prompt-file", required=False, help="JSONL file of prompts")
    check_parser.add_argument("--tier", type=int, default=1, help="Determinism tier")
    check_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    check_parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    check_parser.add_argument("--vary-batch", required=False, help="Comma-separated batch sizes")
    check_parser.add_argument("--mode", choices=["strict", "best-effort"], default="best-effort")
    check_parser.add_argument("--out", required=False, help="Output directory for artifacts")

    subparsers.add_parser("diff", help="Diff traces and emit a report")
    subparsers.add_parser("report", help="Render report artifacts")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "env":
        snapshot = capture_env()
        dump_json(args.out, snapshot)
        return 0

    if args.command == "run":
        if not args.model:
            parser.error("--model is required for run")

        prompts = _load_prompts(args)
        if not prompts:
            parser.error("Prompt input is required via --prompt or --prompt-file")

        os.makedirs(args.out, exist_ok=True)
        env_snapshot = capture_env()
        dump_json(os.path.join(args.out, "env.json"), env_snapshot)

        with DeterministicContext(args.tier, args.mode, args.seed) as ctx:
            backend = _build_backend(args)
            trace_rows = _run_generation(backend, prompts, args)

        dump_json(
            os.path.join(args.out, "determinism_applied.json"),
            _wrap_artifact("determinism_applied", ctx.applied.to_dict()),
        )
        run_config = _build_run_config(
            args,
            env_snapshot.get("device"),
            ctx.applied.tier_effective,
        )
        dump_json(os.path.join(args.out, "run_config.json"), run_config)
        write_trace(os.path.join(args.out, "trace.jsonl"), trace_rows)
        return 0

    if args.command in {"check", "diff", "report"}:
        parser.error("Command not implemented yet. Follow the roadmap in README.md.")

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


def _build_backend(args: argparse.Namespace) -> HFBackend:
    if args.backend != "hf":
        raise ValueError(f"Unsupported backend: {args.backend}")
    return HFBackend(args.model, device=args.device, dtype=args.dtype)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _hash_token_ids(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_generation(backend: HFBackend, prompts: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results = backend.generate(batch, max_new_tokens=args.max_new_tokens, do_sample=False)
        for item in results:
            rows.append(
                {
                    "prompt_id": _hash_prompt(item["prompt"]),
                    "input_token_ids": item["input_ids"],
                    # TODO: Add a privacy mode to store only token hashes/redacted ids.
                    "input_token_ids_hash": _hash_token_ids(item["input_ids"]),
                    "generated_token_ids": item["output_ids"],
                }
            )
    return rows


def _build_run_config(
    args: argparse.Namespace,
    device_snapshot: dict[str, Any] | None,
    tier_effective: int,
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


if __name__ == "__main__":
    raise SystemExit(main())
