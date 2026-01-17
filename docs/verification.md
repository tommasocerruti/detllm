# Verification (local)

This short guide validates that detLLM works end-to-end on CPU using a small HF model.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,hf]'

detllm env --out artifacts/env.json

detllm run --backend hf --model distilgpt2 --prompt "Hello" --tier 1 --batch-size 1 --out artifacts/run1

detllm check --backend hf --model distilgpt2 --prompt "Hello" --tier 1 --runs 3 --batch-size 1 --out artifacts/check1

detllm check --backend hf --model distilgpt2 --prompt "Hello" --tier 1 --runs 3 --batch-size 1 --vary-batch 1,2 --out artifacts/check2
```

## What to expect

After `detllm env`:
- `artifacts/env.json`: environment snapshot with versions, device info, and a fingerprint.

After `detllm run`:
- `artifacts/run1/env.json`: snapshot for the run.
- `artifacts/run1/run_config.json`: inputs + config used for the run.
- `artifacts/run1/determinism_applied.json`: deterministic controls that were applied.
- `artifacts/run1/trace.jsonl`: token trace for the prompt.

After `detllm check`:
- `artifacts/check1/traces/run_0.jsonl` (and run_1/run_2): per-run traces.
- `artifacts/check1/report.json` + `report.txt`: PASS/FAIL with details.
- `artifacts/check1/diffs/first_divergence.json`: only present when a divergence is found.

## Interpreting the report

`report.json` contains:
- `status`: PASS or FAIL.
- `category`: PASS, RUN_VARIANCE_FIXED_BATCH, or BATCH_VARIANCE.
- `details.first_divergence`: where tokens diverged (if any).
- `details.batch_divergence`: which batch size diverged (if any).

If `status` is PASS, the outputs were identical across runs and (if requested) across batch sizes.

## Privacy and schema validation

Redaction flags (CLI):
- `--redact-env` redacts common environment fields.
- `--redact-env-var NAME` redacts specific environment variables (repeatable).

Schema validation (CLI):
- `detllm report --validate-schema` validates `report.json` against the JSON schema.

## Integration test tip

If your shell resolves `pytest` outside the venv, use:

```bash
DETLLM_RUN_INTEGRATION=1 python -m pytest -m integration
```

Or use the Makefile target:

```bash
make test-integration
```

## Tier 2 score capture (verified)

Run a Tier 2 check:

```bash
detllm check --backend hf --model distilgpt2 --prompt "Hello" --tier 2 --runs 2 --batch-size 1 --out artifacts/check_t2
```

Then inspect the trace:

```bash
head -n 1 artifacts/check_t2/traces/run_0.jsonl
```

You should see a `scores` field containing per-token logprobs.
