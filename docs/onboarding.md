# Onboarding Helpers

detLLM includes small commands that reduce setup and artifact-inspection
friction.

## Initialize a local config

```bash
detllm init --out .
```

This writes:

- `detllm.config.json`
- `prompts.jsonl`

Use `--force` to overwrite generated files.

## Check setup

```bash
detllm doctor --config detllm.config.json
```

`doctor` checks Python version, optional extras, schema validation support,
config readability, and artifact directory writability. It does not download
models or run inference.

Use JSON output for scripts:

```bash
detllm doctor --config detllm.config.json --json
```

## Inspect artifacts

```bash
detllm inspect --in artifacts/check1
detllm inspect --in examples/fixtures/phase_demo
```

`inspect` detects common detLLM artifact directories and prints status, key
counts, summaries, and a suggested next command when one is obvious.

## Use profiles

```bash
detllm profile list --config detllm.config.json
detllm profile run quick_check --config detllm.config.json --dry-run
detllm profile run analyze_fixture --config detllm.config.json
```

Profiles reuse existing detLLM commands. Dry runs print the resolved CLI command
without executing it.
