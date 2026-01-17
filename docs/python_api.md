# Python API

The Python API mirrors CLI behavior and returns structured results.

## Quick example

```python
from detllm import check, run

run(
    backend="hf",
    model="distilgpt2",
    prompts=["Hello"],
    tier=1,
    out_dir="artifacts/run1",
)

report = check(
    backend="hf",
    model="distilgpt2",
    prompts=["Hello"],
    runs=3,
    batch_size=1,
    out_dir="artifacts/check1",
)

print(report.status, report.category)
```

## Redaction

```python
from detllm import check

report = check(
    backend="hf",
    model="distilgpt2",
    prompts=["Hello"],
    runs=2,
    redact=True,
    redact_env_vars=["CUBLAS_WORKSPACE_CONFIG"],
    out_dir="artifacts/check_redacted",
)
```

## Schema validation

```python
from detllm import check

report = check(
    backend="hf",
    model="distilgpt2",
    prompts=["Hello"],
    runs=2,
    validate_schema=True,
    out_dir="artifacts/check_validated",
)
```
