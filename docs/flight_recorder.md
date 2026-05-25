# Flight Recorder

Flight Recorder turns detLLM repro packs into local debugging reports for ML
engineers and researchers. It ranks likely causes and can run targeted replay
probes when prompt text was captured explicitly.

## Diagnose a repro pack

```bash
detllm diagnose --in artifacts/check1
```

This writes:

- `artifacts/check1/diagnosis.json`
- `artifacts/check1/diagnosis.txt`

The diagnosis artifact contains `status`, `category`, `summary`, ranked
`causes`, supporting `evidence`, and a `probe_plan`.

## Replay probes

```bash
detllm replay --in artifacts/check1 --probe auto
```

`auto` runs only probes backed by the diagnosis. Available probes are:

- `batch-shape`
- `isolate-prompts`
- `score-margins`
- `tier2`

Replay requires prompt text. By default, detLLM traces remain token-id based.
Capture prompt text only for local debugging:

```bash
detllm check --backend hf --model distilgpt2 \
  --prompt "Hello" \
  --runs 3 \
  --include-token-text \
  --out artifacts/check_replayable
```

## Score margins

Top-k score capture is optional and disabled by default:

```bash
detllm check --backend hf --model distilgpt2 \
  --prompt "Hello" \
  --tier 2 \
  --capture-topk-scores 5 \
  --out artifacts/check_margins
```

When available, diagnosis can flag fragile near-ties with
`score_margin_instability`.
