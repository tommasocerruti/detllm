# Changelog

## Unreleased

- Flight Recorder: `diagnose` and `replay` commands, Python API parity, diagnosis
  artifacts, targeted replay probes, and optional top-k score capture.
- Reproducibility phase diagrams: `phase` command and Python API for sweeping
  batch size, dtype, and max token settings into stable/fragile/unstable maps.
- Statistical reproducibility analysis: `analyze` command and Python API for
  Wilson confidence intervals, risk scoring, and next-experiment recommendations.
- Adaptive experiment planner: `recommend` command and Python API for ranked,
  recommend-only follow-up phase experiments.
- Runnable examples: guided local workflows plus checked prompt and phase
  fixtures for `check`, `diagnose`, `replay`, `phase`, `analyze`, and `recommend`.
- Onboarding helpers: `init`, `doctor`, `inspect`, and `profile` commands for
  setup checks, starter configs, artifact summaries, and reusable workflows.

## 0.1.1

- docs: fix README assets for PyPI (absolute URLs).

## 0.1.0

- CLI: `env`, `run`, `check`, `diff`, `report` commands.
- Artifacts: env snapshot, run config, determinism applied, trace, report.
- Determinism tiers, diffing, and batch variance reporting.
- HF backend (CPU-first) and vLLM Tier 0 adapter.
- Python API (`run`, `check`) and schema validation.
- Integration tests with tiny HF models.
