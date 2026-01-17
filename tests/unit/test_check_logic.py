import argparse
import pytest

from detllm.cli import main as cli_main
from detllm.diff.diff import DiffResult


def test_parse_vary_batch_empty():
    assert cli_main._parse_vary_batch(None) == []
    assert cli_main._parse_vary_batch("") == []


def test_parse_vary_batch_values():
    assert cli_main._parse_vary_batch("1,2,4") == [1, 2, 4]


def test_parse_vary_batch_rejects_non_positive():
    with pytest.raises(ValueError):
        cli_main._parse_vary_batch("0")


def test_report_category_prefers_run_variance():
    run_result = DiffResult(status="FAIL", category="RUN_VARIANCE_FIXED_BATCH", first_divergence={})
    batch_result = DiffResult(status="FAIL", category="RUN_VARIANCE_FIXED_BATCH", first_divergence={})
    assert cli_main._report_category(run_result, batch_result) == "RUN_VARIANCE_FIXED_BATCH"


def test_report_category_batch_variance_only():
    run_result = DiffResult(status="PASS", category="PASS", first_divergence=None)
    batch_result = DiffResult(status="FAIL", category="RUN_VARIANCE_FIXED_BATCH", first_divergence={})
    assert cli_main._report_category(run_result, batch_result) == "BATCH_VARIANCE"


def test_clone_args_updates_batch_size():
    args = argparse.Namespace(batch_size=2, other="x")
    clone = cli_main._clone_args(args, batch_size=4)
    assert clone.batch_size == 4
    assert clone.other == "x"


def test_batch_divergence_detail_reports_first_batch():
    run_result = DiffResult(status="PASS", category="PASS", first_divergence=None)
    batch_diffs = [
        (1, DiffResult(status="PASS", category="PASS", first_divergence=None)),
        (2, DiffResult(status="FAIL", category="RUN_VARIANCE_FIXED_BATCH", first_divergence={"idx": 1})),
    ]
    detail = cli_main._batch_divergence_detail(batch_diffs, run_result)
    assert detail["batch_size"] == 2
