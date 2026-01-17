from detllm.diff.diff import diff_traces


def test_diff_traces_score_divergence():
    base = [{"prompt_id": "a", "generated_token_ids": [1], "scores": [0.1, 0.2]}]
    other = [{"prompt_id": "a", "generated_token_ids": [1], "scores": [0.1, 0.3]}]
    result = diff_traces(base, other)
    assert result.status == "FAIL"
    assert result.category == "SCORE_VARIANCE"
