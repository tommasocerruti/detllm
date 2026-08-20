# Reproducibility Phase Diagram

`detllm phase` runs a local sweep over controlled inference variables and
classifies each configuration as `stable`, `fragile`, or `unstable`.

V1 supports these axes:

- `batch_size`
- `dtype`
- `max_new_tokens`

## Example

```bash
detllm phase \
  --backend hf \
  --model distilgpt2 \
  --prompt-file prompts.jsonl \
  --axis batch_size=1,2,4 \
  --axis dtype=float32,bfloat16 \
  --axis max_new_tokens=16,32 \
  --runs 5 \
  --tier 2 \
  --capture-topk-scores 5 \
  --max-cells 12 \
  --out artifacts/phase/distilgpt2
```

Use `--dry-run` to write the planned grid without model inference.

## Artifacts

The top-level output directory contains:

- `phase_diagram.json`: structured experiment result.
- `phase_diagram.csv`: one row per cell for plotting.
- `phase_diagram.txt`: text summary.
- `cells/<cell_id>/`: normal detLLM check artifacts plus Flight Recorder diagnosis.

## Classification

- `stable`: the cell passes and has no score-margin instability diagnosis.
- `fragile`: the cell passes, but diagnosis identifies fragile top-k margins.
- `unstable`: the cell fails with a hard divergence or unsupported guarantee.
