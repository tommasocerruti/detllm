from detllm.diff.diff import aggregate_diffs, diff_traces, first_token_divergence


def test_first_token_divergence():
    assert first_token_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_token_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_token_divergence([1], [1, 2]) == 1


def test_diff_traces_pass():
    base = [
        {"prompt_id": "a", "generated_token_ids": [1, 2]},
        {"prompt_id": "b", "generated_token_ids": [3]},
    ]
    other = [
        {"prompt_id": "a", "generated_token_ids": [1, 2]},
        {"prompt_id": "b", "generated_token_ids": [3]},
    ]
    result = diff_traces(base, other)
    assert result.status == "PASS"


def test_diff_traces_detects_divergence():
    base = [{"prompt_id": "a", "generated_token_ids": [1, 2]}]
    other = [{"prompt_id": "a", "generated_token_ids": [1, 9]}]
    result = diff_traces(base, other)
    assert result.status == "FAIL"
    assert result.category == "RUN_VARIANCE_FIXED_BATCH"


def test_aggregate_diffs_returns_first_failure():
    pass_result = diff_traces(
        [{"prompt_id": "a", "generated_token_ids": [1]}],
        [{"prompt_id": "a", "generated_token_ids": [1]}],
    )
    fail_result = diff_traces(
        [{"prompt_id": "a", "generated_token_ids": [1]}],
        [{"prompt_id": "a", "generated_token_ids": [2]}],
    )
    result = aggregate_diffs([pass_result, fail_result])
    assert result.status == "FAIL"
