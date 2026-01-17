"""Trace diffing and classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DiffResult:
    status: str
    category: str
    first_divergence: dict[str, Any] | None


def first_token_divergence(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def diff_traces(
    base: list[dict[str, Any]],
    other: list[dict[str, Any]],
) -> DiffResult:
    if len(base) != len(other):
        return DiffResult(
            status="FAIL",
            category="GEN_CONTEXT_MISMATCH",
            first_divergence={"reason": "trace lengths differ"},
        )

    for idx, (left, right) in enumerate(zip(base, other)):
        if left.get("prompt_id") != right.get("prompt_id"):
            return DiffResult(
                status="FAIL",
                category="GEN_CONTEXT_MISMATCH",
                first_divergence={
                    "index": idx,
                    "reason": "prompt_id mismatch",
                    "left_prompt_id": left.get("prompt_id"),
                    "right_prompt_id": right.get("prompt_id"),
                },
            )

        divergence = first_token_divergence(
            left.get("generated_token_ids", []),
            right.get("generated_token_ids", []),
        )
        if divergence is not None:
            return DiffResult(
                status="FAIL",
                category="RUN_VARIANCE_FIXED_BATCH",
                first_divergence={
                    "index": idx,
                    "token_index": divergence,
                    "left_token": _safe_token(left, divergence),
                    "right_token": _safe_token(right, divergence),
                },
            )

        score_divergence = _first_score_divergence(left.get("scores"), right.get("scores"))
        if score_divergence is not None:
            return DiffResult(
                status="FAIL",
                category="SCORE_VARIANCE",
                first_divergence={
                    "index": idx,
                    "score_index": score_divergence,
                },
            )

    return DiffResult(status="PASS", category="PASS", first_divergence=None)


def _safe_token(row: dict[str, Any], index: int) -> int | None:
    tokens = row.get("generated_token_ids", [])
    if index < len(tokens):
        return tokens[index]
    return None


def _first_score_divergence(
    left_scores: list[float] | None,
    right_scores: list[float] | None,
) -> int | None:
    if left_scores is None or right_scores is None:
        return None
    n = min(len(left_scores), len(right_scores))
    for i in range(n):
        if left_scores[i] != right_scores[i]:
            return i
    if len(left_scores) != len(right_scores):
        return n
    return None


def aggregate_diffs(diffs: Iterable[DiffResult]) -> DiffResult:
    for diff in diffs:
        if diff.status != "PASS":
            return diff
    return DiffResult(status="PASS", category="PASS", first_divergence=None)
