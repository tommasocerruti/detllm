# Determinism boundary

This document describes determinism boundaries for DetLLM tiers and why GPU determinism is conditional.

## Tier 1: fixed batch only

Tier 1 guarantees repeatability only for a fixed batch size. If batch size changes, outputs may change even under greedy decoding.

Why:
- GPU math is not strictly associative, so small scheduling differences can alter results.
- Batching affects kernel selection and numerics.

## Tier 2: scores/logprobs

Tier 2 extends Tier 1 by requiring score/logprob equality. This is capability-gated:
- Some backends do not expose stable per-token scores.
- Some backends expose scores but do not guarantee stability across runs.

If scores are unavailable, strict Tier 2 fails with `UNSUPPORTED_REQUEST`. Best-effort downgrades to Tier 1.

## GPU determinism is conditional

Deterministic algorithms are not always available for every op, and some require environment variables to be set before process start.
