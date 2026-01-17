from detllm.backends.vllm import VLLMBackend


def test_vllm_backend_capabilities():
    backend = VLLMBackend("fake-model")
    caps = backend.capabilities()
    assert caps.supports_tier1_fixed_batch is False
    assert caps.supports_scores is False
    assert caps.supports_torch_deterministic is False
