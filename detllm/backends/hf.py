"""Hugging Face Transformers backend adapter (CPU-first)."""

from __future__ import annotations

from typing import Any

from detllm.backends.base import BackendAdapter, BackendCapabilities
from detllm.logging import get_logger

logger = get_logger("backends.hf")


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
            from transformers.utils import logging as hf_logging
        except Exception as exc:
            raise RuntimeError(
                "HF backend requires torch and transformers. Install detllm with the"
                " required dependencies."
            ) from exc

        hf_logging.set_verbosity_error()
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._tokenizer_id = getattr(self.tokenizer, "name_or_path", self.model_id)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            # TODO: Allow configuring pad token instead of defaulting to EOS.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.debug("Set pad_token_id to eos_token_id=%s", self.tokenizer.pad_token_id)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_tier1_fixed_batch=True,
            supports_scores=True,
            supports_torch_deterministic=True,
            # TODO: Add score/logprob capture support for Tier 2 when available.
            notes=["CPU-only deterministic controls are best-effort."],
        )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 32,
        do_sample: bool = False,
        capture_scores: bool = False,
    ) -> list[dict[str, Any]]:
        import torch
        import torch.nn.functional as torch_f

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("HF backend not initialized.")

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                output_scores=capture_scores,
                return_dict_in_generate=capture_scores,
            )

        if capture_scores:
            sequences = outputs.sequences
            score_tensors = outputs.scores
        else:
            sequences = outputs
            score_tensors = None

        results: list[dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            scores = None
            if capture_scores and score_tensors is not None:
                input_len = int(inputs["attention_mask"][i].sum().item())
                scores = _token_logprobs(
                    sequences[i],
                    input_len,
                    score_tensors,
                    i,
                    torch_f,
                )
            results.append(
                {
                    "prompt": prompt,
                    "input_ids": inputs["input_ids"][i].tolist(),
                    "output_ids": sequences[i].tolist(),
                    "scores": scores,
                    "tokenizer_id": self._tokenizer_id,
                }
            )
        return results


def _token_logprobs(
    sequence: torch.Tensor,
    input_len: int,
    score_tensors: list[torch.Tensor],
    batch_index: int,
    torch_f,
) -> list[float]:
    logprobs: list[float] = []
    for step, scores in enumerate(score_tensors):
        token_id = int(sequence[input_len + step].item())
        log_probs = torch_f.log_softmax(scores[batch_index], dim=-1)
        logprobs.append(float(log_probs[token_id].item()))
    return logprobs
