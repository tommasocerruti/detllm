# detLLM Examples

This directory contains small, local-safe inputs for the walkthroughs in
`docs/examples.md`.

- `prompts/basic.jsonl`: tiny prompts for first checks.
- `prompts/research_debugging.jsonl`: prompts shaped like local ML debugging
  probes.
- `fixtures/phase_demo/phase_diagram.json`: synthetic phase diagram used for
  analysis and recommendation examples without running model inference.

The fixture is intentionally small and synthetic. It exists so contributors can
verify examples quickly on a laptop or in unit tests.
