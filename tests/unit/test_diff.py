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


def test_diff_traces_detects_tokenization_mismatch():
    base = [{"prompt_id": "a", "input_token_ids_hash": "x", "generated_token_ids": [1]}]
    other = [{"prompt_id": "a", "input_token_ids_hash": "y", "generated_token_ids": [1]}]
    result = diff_traces(base, other)
    assert result.status == "FAIL"
    assert result.category == "TOKENIZATION_MISMATCH"


def test_diff_traces_detects_tokenizer_id_mismatch():
    base = [{"prompt_id": "a", "tokenizer_id": "tok1", "generated_token_ids": [1]}]
    other = [{"prompt_id": "a", "tokenizer_id": "tok2", "generated_token_ids": [1]}]
    result = diff_traces(base, other)
    assert result.status == "FAIL"
    assert result.category == "TOKENIZATION_MISMATCH"


def test_diff_traces_detects_decoding_mismatch():
    base = [{"prompt_id": "a", "decoding_max_new_tokens": 10, "generated_token_ids": [1]}]
    other = [{"prompt_id": "a", "decoding_max_new_tokens": 12, "generated_token_ids": [1]}]
    result = diff_traces(base, other)
    assert result.status == "FAIL"
    assert result.category == "GEN_CONTEXT_MISMATCH"


def test_diff_traces_detects_temperature_mismatch():
    base = [{"prompt_id": "a", "decoding_temperature": 0.0, "generated_token_ids": [1]}]
    other = [{"prompt_id": "a", "decoding_temperature": 0.5, "generated_token_ids": [1]}]
    result = diff_traces(base, other)
    assert result.status == "FAIL"
    assert result.category == "GEN_CONTEXT_MISMATCH"


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


def test_diff_traces_length_mismatch_details():
    result = diff_traces(
        [{"prompt_id": "a", "generated_token_ids": [1]}],
        [],
    )
    assert result.category == "GEN_CONTEXT_MISMATCH"
    assert result.first_divergence["left_len"] == 1
