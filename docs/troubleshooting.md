# Troubleshooting

This document maps each failure category to likely causes and next actions.

## RUN_VARIANCE_FIXED_BATCH

Likely causes:
- Nondeterministic ops or backend defaults.
- Subtle environment differences across runs.

Next actions:
- Use `--mode strict` to surface unmet requirements.
- Check `determinism_applied.json` for downgrades.

## BATCH_VARIANCE

Likely causes:
- Batch size changes kernel choices or numerics.

Next actions:
- Treat batch invariance as a separate requirement; compare outputs per batch size.

## SCORE_VARIANCE

Likely causes:
- Backend scores are not stable or are computed differently across runs.

Next actions:
- Ensure Tier 2 is supported; consider downgrading to Tier 1 if scores are unreliable.

## UNSUPPORTED_REQUEST

Likely causes:
- Requested tier requires capabilities the backend does not support.

Next actions:
- Switch to best-effort mode or use a backend with stronger determinism support.

## Redaction and schema validation

If artifacts contain sensitive fields:
- Use `--redact-env` or `--redact-env-var` when running `env`, `run`, or `check`.

If schema validation fails:
- Install the optional `schema` extra and rerun `detllm report --validate-schema`.

## Pytest uses system Python

If `pytest` resolves to your system Python instead of the venv, prefer:
- `python -m pytest` (uses the active interpreter)
