# Statistical Reproducibility Analysis

`detllm analyze` turns a `phase_diagram.json` into cell-level reproducibility
risk estimates with uncertainty.

```bash
detllm analyze \
  --in artifacts/phase/distilgpt2 \
  --out artifacts/phase/distilgpt2/analysis \
  --confidence 0.95
```

The input directory must contain `phase_diagram.json`. The output directory
defaults to `<phase_dir>/analysis`.

## Outputs

- `analysis.json`: structured summary, intervals, risk score, recommendations,
  and derived cell rows.
- `analysis.csv`: one row per phase cell for plotting or spreadsheet analysis.
- `analysis.txt`: compact scientific summary for local debugging notes.

## Metrics

- `unstable_rate`: unstable executed cells divided by executed cells.
- `nonstable_rate`: fragile plus unstable executed cells divided by executed
  cells.
- `risk_score`: `100 * (unstable + 0.5 * fragile) / executed`.

Wilson confidence intervals are reported for unstable and non-stable rates.
Supported confidence levels are `0.90`, `0.95`, and `0.99`.

When no unstable cells are observed, the text summary includes rule-of-three
guidance: with `N` executed cells, the 95% upper bound is approximately `3/N`.

## Recommendations

The analyzer suggests focused next experiments when:

- intervals are wide, indicating more runs or cells are needed;
- fragile cells exist, indicating score-margin probes are useful;
- unstable cells cluster around a specific axis value;
- skipped phase cells remain.
