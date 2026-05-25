import argparse

from detllm.cli import main as cli_main


class TopKBackend:
    def __init__(self):
        self.kwargs = None

    def generate(self, prompts, **kwargs):
        self.kwargs = kwargs
        return [
            {
                "prompt": prompts[0],
                "input_ids": [1],
                "output_ids": [1, 2],
                "scores": [-0.1],
                "topk_token_ids": [[2, 3]],
                "topk_scores": [[-0.1, -0.2]],
                "tokenizer_id": "fake-tokenizer",
            }
        ]


def test_run_generation_passes_and_records_topk_scores():
    backend = TopKBackend()
    args = argparse.Namespace(
        batch_size=1,
        max_new_tokens=1,
        model="fake",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        capture_topk_scores=2,
    )

    rows = cli_main._run_generation(
        backend,
        ["hello"],
        args,
        capture_scores=True,
        capture_topk_scores=2,
    )

    assert backend.kwargs["capture_topk_scores"] == 2
    assert rows[0]["topk_token_ids"] == [[2, 3]]
    assert rows[0]["topk_scores"] == [[-0.1, -0.2]]


def test_trace_model_accepts_topk_fields():
    rows = cli_main._coerce_trace_rows(
        [
            {
                "prompt_id": "p0",
                "generated_token_ids": [2],
                "topk_token_ids": [[2, 3]],
                "topk_scores": [[-0.1, -0.2]],
            }
        ]
    )

    assert rows[0]["topk_token_ids"] == [[2, 3]]
