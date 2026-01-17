# vLLM backend

detLLM's vLLM adapter is Tier 0 only. It captures outputs for measurement but does not claim fixed-batch repeatability or score stability.

Notes:
- Batch invariance is not guaranteed by default in vLLM.
- Determinism controls are limited by backend settings and deployment configuration.

If you need Tier 1/2 guarantees, use the HF backend or another backend that exposes the required controls.
