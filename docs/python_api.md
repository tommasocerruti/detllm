# Python API

The Python API mirrors CLI behavior and returns structured results.

## Quick example

```python
from detllm import check, diagnose, replay, run

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

diagnosis = diagnose("artifacts/check1")
print(diagnosis.summary)
```

## API reference (minimal)

```python
from detllm import check, diagnose, replay, run

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
    capture_topk_scores: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
    out_dir: str = "artifacts/run",
    redact: bool = False,
    redact_env_vars: list[str] | None = None,
    validate_schema: bool = False,
    include_token_text: bool = False,
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
    capture_topk_scores: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
    out_dir: str = "artifacts/check",
    redact: bool = False,
    redact_env_vars: list[str] | None = None,
    validate_schema: bool = False,
    include_token_text: bool = False,
)

# diagnose(...)
# Returns: Diagnosis(status, category, summary, causes, evidence, probe_plan, out_dir)
diagnose(
    in_dir: str,
    out_dir: str | None = None,
    include_token_text: bool = False,
    validate_schema: bool = False,
)

# replay(...)
# Returns: ReplayResult(status, out_dir, executed_probes, skipped_probes)
replay(
    in_dir: str,
    probe: str = "auto",
    out_dir: str | None = None,
    include_token_text: bool = False,
    capture_topk_scores: int = 5,
    validate_schema: bool = False,
    backend_adapter: BackendAdapter | None = None,
)
```

Replay requires prompt text in trace rows. Use `include_token_text=True` when
creating local debugging packs that you intend to replay.

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
