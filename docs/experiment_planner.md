# Adaptive Experiment Planner

`detllm recommend` consumes a phase diagram and produces a ranked plan for the
next local experiments to run.

```bash
detllm recommend \
  --in artifacts/phase/distilgpt2 \
  --out artifacts/phase/distilgpt2/recommendations \
  --budget-cells 8 \
  --strategy auto \
  --confidence 0.95
```

The input directory must contain `phase_diagram.json`. If
`analysis/analysis.json` is present, the planner uses it. Otherwise it computes
the same analysis in memory without writing analysis artifacts.

## Outputs

- `experiment_plan.json`: structured ranked follow-up experiments.
- `experiment_plan.csv`: one row per recommendation.
- `experiment_plan.txt`: compact local-debugging summary.

Each recommendation includes a rank, type, score, axes, reason, source cell ids,
suggested overrides, and a command hint.

## Strategies

- `auto`: combines coverage, fragility, cluster, boundary, and uncertainty
  heuristics.
- `coverage`: prioritizes skipped or planned cells.
- `fragility`: reruns fragile cells with deeper top-k capture.
- `boundary`: probes numeric class transitions along `batch_size` and
  `max_new_tokens`.

V1 is recommend-only. It never loads a model, calls a backend, or runs inference.
