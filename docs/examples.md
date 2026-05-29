# Examples

These examples are designed for local ML debugging. They use small prompt files
and explicit output directories so the artifacts are easy to inspect and delete.

Install with the Hugging Face extra if you want to run the live model commands:

```bash
pip install 'detllm[hf]'
```

The analysis and recommendation examples use the tracked fixture in
`examples/fixtures/phase_demo`, so they do not download a model or run
inference.

## Start with a local config

Create a starter config and prompt file:

```bash
detllm init --out .
```

Check the local setup without downloading models:

```bash
detllm doctor --config detllm.config.json
```

Preview a configured workflow:

```bash
detllm profile list --config detllm.config.json
detllm profile run quick_check --config detllm.config.json --dry-run
```

After running any example, summarize the artifact directory:

```bash
detllm inspect --in examples/fixtures/phase_demo
```

## First reproducibility check

Run a fixed-batch repeatability check:

```bash
detllm check \
  --backend hf \
  --model distilgpt2 \
  --prompt-file examples/prompts/basic.jsonl \
  --tier 1 \
  --runs 3 \
  --batch-size 1 \
  --out artifacts/examples/check_basic
```

Inspect:

- `artifacts/examples/check_basic/report.txt`
- `artifacts/examples/check_basic/report.json`
- `artifacts/examples/check_basic/traces/run_0.jsonl`

`PASS` means the checked runs matched under the requested guarantee. `FAIL`
means detLLM found a concrete divergence and wrote the first divergence into the
artifact pack.

## Debug a divergence with Flight Recorder

Create a replayable check pack. Prompt text is opt-in because default traces are
token-id based:

```bash
detllm check \
  --backend hf \
  --model distilgpt2 \
  --prompt-file examples/prompts/research_debugging.jsonl \
  --tier 2 \
  --runs 3 \
  --batch-size 1 \
  --vary-batch 1,2 \
  --capture-topk-scores 5 \
  --include-token-text \
  --out artifacts/examples/check_replayable
```

Diagnose the pack:

```bash
detllm diagnose \
  --in artifacts/examples/check_replayable
```

Run targeted probes chosen from the diagnosis:

```bash
detllm replay \
  --in artifacts/examples/check_replayable \
  --probe auto \
  --out artifacts/examples/check_replayable/replay
```

Inspect:

- `diagnosis.txt` for ranked causes
- `diagnosis.json` for evidence and probe plans
- `replay/replay.txt` and `replay/replay.json` for executed probes

Common causes include batch-shape drift, environment drift, tokenization drift,
unsupported guarantees, and score-margin instability.

## Build a phase diagram

Sweep a small grid of controlled inference variables:

```bash
detllm phase \
  --backend hf \
  --model distilgpt2 \
  --prompt-file examples/prompts/research_debugging.jsonl \
  --axis batch_size=1,2 \
  --axis dtype=float32 \
  --axis max_new_tokens=16,32 \
  --runs 3 \
  --tier 2 \
  --capture-topk-scores 5 \
  --max-cells 4 \
  --out artifacts/examples/phase_demo
```

Inspect:

- `artifacts/examples/phase_demo/phase_diagram.txt`
- `artifacts/examples/phase_demo/phase_diagram.csv`
- `artifacts/examples/phase_demo/cells/<cell_id>/report.txt`

Cell classes:

- `stable`: the cell passed and no score-margin instability was detected.
- `fragile`: the cell passed, but near-tie score behavior suggests risk.
- `unstable`: the cell failed with hard divergence or an unsupported guarantee.

Use `--dry-run` with the same command to preview the grid without inference.

## Estimate risk with statistical analysis

Use the tracked fixture for a no-inference analysis example:

```bash
detllm analyze \
  --in examples/fixtures/phase_demo \
  --out artifacts/examples/phase_demo/analysis \
  --confidence 0.95
```

Inspect:

- `artifacts/examples/phase_demo/analysis/analysis.txt`
- `artifacts/examples/phase_demo/analysis/analysis.json`
- `artifacts/examples/phase_demo/analysis/analysis.csv`

The report includes unstable and non-stable rates, Wilson confidence intervals,
a reproducibility risk score, and recommendations such as increasing sample
size or rerunning fragile cells with deeper top-k capture.

## Choose next experiments with recommendations

Use the same fixture to rank follow-up cells:

```bash
detllm recommend \
  --in examples/fixtures/phase_demo \
  --out artifacts/examples/phase_demo/recommendations \
  --budget-cells 8 \
  --strategy auto \
  --confidence 0.95
```

Inspect:

- `artifacts/examples/phase_demo/recommendations/experiment_plan.txt`
- `artifacts/examples/phase_demo/recommendations/experiment_plan.json`
- `artifacts/examples/phase_demo/recommendations/experiment_plan.csv`

The planner is recommend-only. It does not load a model or run inference. It
ranks skipped cells, fragile cells, unstable clusters, boundary probes, and
uncertainty-reducing reruns.

## Python API equivalent workflow

```python
from detllm import analyze, check, diagnose, phase, recommend, replay

report = check(
    backend="hf",
    model="distilgpt2",
    prompts=["Choose one: A or B. Answer with a single letter."],
    tier=1,
    runs=3,
    batch_size=1,
    out_dir="artifacts/examples/check_basic_py",
)
print(report.status, report.category)

diagnosis = diagnose("artifacts/examples/check_basic_py")
print(diagnosis.summary)

diagram = phase(
    backend="hf",
    model="distilgpt2",
    prompts=["Classify this run as stable or unstable."],
    axes={
        "batch_size": [1, 2],
        "dtype": ["float32"],
        "max_new_tokens": [16],
    },
    runs=3,
    tier=1,
    max_cells=2,
    out_dir="artifacts/examples/phase_demo_py",
)
print(diagram.summary())

analysis = analyze(
    "examples/fixtures/phase_demo",
    out_dir="artifacts/examples/phase_demo_py/analysis",
)
print(analysis.risk_score)

plan = recommend(
    "examples/fixtures/phase_demo",
    out_dir="artifacts/examples/phase_demo_py/recommendations",
    budget_cells=4,
)
print(plan.recommendations[0].recommendation_type)
```

Use `replay(...)` on packs created with `include_token_text=True`; otherwise the
replay step intentionally skips probes that require original prompt text.
