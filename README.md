# DetLLM
DetLLM helps you verify whether an LLM inference setup is reproducible and gives you a minimal repro pack when it is not.

DetLLM measures variance across repeated runs and across batch sizes, and explains divergences without overpromising.

## Quickstart

```bash
pip install -e .
detllm check --backend hf --model <model_id> --prompt "Hello" --tier 1 --runs 5 --batch-size 1
```

## Tiers

- Tier 0: artifacts + deterministic diff/report (no equality guarantees)
- Tier 1: repeatability across runs for a fixed batch size
- Tier 2: Tier 1 + score/logprob equality (capability-gated)

Tier 1 guarantees repeatability only for a fixed batch size; batch invariance is measured separately.

## Artifacts (minimal repro pack)

Each run writes an `artifacts/<run_id>/` folder:

- `env.json`
- `run_config.json`
- `determinism_applied.json`
- `trace.jsonl`
- `report.json` + `report.txt`
- `diffs/first_divergence.json`

## CLI (planned)

- `detllm env`
- `detllm run`
- `detllm check`
- `detllm diff`
- `detllm report`

## Project status

This repo is a WIP skeleton for the Milestone 0 deliverables:

- Packaging + CLI entrypoint
- CI + tests
- Base docs (LICENSE, CONTRIBUTING, CODE_OF_CONDUCT)
