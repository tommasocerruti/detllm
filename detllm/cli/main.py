"""DetLLM command-line interface."""

from __future__ import annotations

import argparse
import sys

from detllm.core.artifacts import dump_json
from detllm.core.env import capture_env


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
    run_parser.add_argument("--mode", choices=["strict", "best-effort"], default="best-effort")
    run_parser.add_argument("--out", required=False, help="Output directory for artifacts")

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

    if args.command in {"run", "check", "diff", "report"}:
        parser.error("Command not implemented yet. Follow the roadmap in README.md.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
