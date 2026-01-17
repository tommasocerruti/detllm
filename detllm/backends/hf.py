"""Hugging Face Transformers backend adapter (CPU-first)."""

from __future__ import annotations

from typing import Any

from detllm.backends.base import BackendAdapter, BackendCapabilities


class HFBackend(BackendAdapter):
    def __init__(self, model_id: str, device: str = "cpu", dtype: str = "float32"):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "HF backend requires torch and transformers. Install detllm with the"
                " required dependencies."
            ) from exc

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            # TODO: Allow configuring pad token instead of defaulting to EOS.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_tier1_fixed_batch=True,
            supports_scores=False,
            supports_torch_deterministic=True,
            notes=["CPU-only deterministic controls are best-effort."],
        )

    def generate(
        self, prompts: list[str], max_new_tokens: int = 32, do_sample: bool = False
    ) -> list[dict[str, Any]]:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("HF backend not initialized.")

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        results: list[dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            results.append(
                {
                    "prompt": prompt,
                    "input_ids": inputs["input_ids"][i].tolist(),
                    "output_ids": outputs[i].tolist(),
                }
            )
        return results
