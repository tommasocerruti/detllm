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

## API reference (minimal)

```python
from detllm import run, check

# run(...)
# Returns: RunResult(status: str, category: str, out_dir: str)
run(
    backend: str,
    model: str,
    prompts: list[str],
    tier: int = 1,
    mode: str = "best-effort",
    batch_size: int = 1,
    seed: int = 0,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
    out_dir: str = "artifacts/run",
    redact: bool = False,
    redact_env_vars: list[str] | None = None,
    validate_schema: bool = False,
)

# check(...)
# Returns: Report(status: str, category: str, details: dict)
check(
    backend: str,
    model: str,
    prompts: list[str],
    tier: int = 1,
    mode: str = "best-effort",
    runs: int = 3,
    batch_size: int = 1,
    vary_batch: list[int] | None = None,
    seed: int = 0,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
    out_dir: str = "artifacts/check",
    redact: bool = False,
    redact_env_vars: list[str] | None = None,
    validate_schema: bool = False,
)
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
