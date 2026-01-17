"""vLLM backend adapter (Tier 0 measurement only)."""

from __future__ import annotations

from typing import Any

from detllm.backends.base import BackendAdapter, BackendCapabilities


class VLLMBackend(BackendAdapter):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._llm = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_tier1_fixed_batch=False,
            supports_scores=False,
            supports_torch_deterministic=False,
            notes=[
                "Tier 0 measurement only; batch invariance not guaranteed.",
                "Determinism controls are limited by backend settings.",
            ],
        )

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM
        except Exception as exc:
            raise RuntimeError(
                "vLLM backend requires vllm. Install detllm with the vllm extra."
            ) from exc
        self._llm = LLM(model=self.model_id)

    def generate(self, prompts: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        self._ensure_loaded()
        try:
            from vllm import SamplingParams
        except Exception as exc:
            raise RuntimeError("vLLM SamplingParams unavailable.") from exc

        max_new_tokens = int(kwargs.get("max_new_tokens", 32))
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        outputs = self._llm.generate(prompts, params)

        results: list[dict[str, Any]] = []
        for prompt, output in zip(prompts, outputs):
            token_ids = output.outputs[0].token_ids
            results.append(
                {
                    "prompt": prompt,
                    "input_ids": [],
                    # TODO: Include input token ids when vLLM exposes tokenizer hooks.
                    "output_ids": token_ids,
                }
            )
        return results
