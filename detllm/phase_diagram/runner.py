"""Phase diagram execution."""

from __future__ import annotations

import os
from typing import Any, Sequence

from detllm.backends.base import BackendAdapter
from detllm.core.artifacts import dump_json, load_json, validate_artifact
from detllm.flight_recorder.diagnose import diagnose_directory
from detllm.phase_diagram.phase import (
    PhaseDiagram,
    cell_artifact_path,
    classify_cell,
    expand_grid,
    phase_from_cells,
)
from detllm.phase_diagram.render import render_phase_text, write_phase_csv
from detllm.trace.io import read_trace


def run_phase(
    *,
    backend: str,
    model: str,
    prompts: Sequence[str],
    axes: dict[str, list[Any]],
    runs: int = 3,
    tier: int = 1,
    mode: str = "best-effort",
    seed: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
    capture_topk_scores: int = 0,
    out_dir: str = "artifacts/phase",
    max_cells: int | None = None,
    dry_run: bool = False,
    validate_schema: bool = False,
    include_token_text: bool = False,
    backend_adapter: BackendAdapter | None = None,
) -> PhaseDiagram:
    os.makedirs(out_dir, exist_ok=True)
    cells = expand_grid(axes, max_cells=max_cells)
    diagram = phase_from_cells(cells, axes=axes, out_dir=out_dir, dry_run=dry_run)

    if not dry_run:
        for cell in cells:
            if cell.execution_status == "skipped":
                continue
            _execute_cell(
                cell,
                backend=backend,
                model=model,
                prompts=prompts,
                runs=runs,
                tier=tier,
                mode=mode,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                device=device,
                capture_topk_scores=capture_topk_scores,
                out_dir=out_dir,
                validate_schema=validate_schema,
                include_token_text=include_token_text,
                backend_adapter=backend_adapter,
            )

    _write_phase_artifacts(diagram, validate_schema=validate_schema)
    return diagram


def _execute_cell(
    cell,
    *,
    backend: str,
    model: str,
    prompts: Sequence[str],
    runs: int,
    tier: int,
    mode: str,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    capture_topk_scores: int,
    out_dir: str,
    validate_schema: bool,
    include_token_text: bool,
    backend_adapter: BackendAdapter | None,
) -> None:
    from detllm.api import check

    cell_dir = cell_artifact_path(out_dir, cell)
    report = check(
        backend=backend,
        model=model,
        prompts=prompts,
        tier=tier,
        mode=mode,
        runs=runs,
        batch_size=int(cell.axes.get("batch_size", 1)),
        seed=seed,
        max_new_tokens=int(cell.axes.get("max_new_tokens", 32)),
        capture_topk_scores=capture_topk_scores,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        device=device,
        dtype=str(cell.axes.get("dtype", "float32")),
        out_dir=cell_dir,
        backend_adapter=backend_adapter,
        validate_schema=validate_schema,
        include_token_text=include_token_text,
    )
    diagnosis = diagnose_directory(cell_dir, validate_schema=validate_schema)
    diagnosis_payload = load_json(os.path.join(cell_dir, "diagnosis.json"))
    cell.execution_status = "executed"
    cell.classification = classify_cell(report.to_dict(), diagnosis_payload)
    cell.report_status = report.status
    cell.report_category = report.category
    cell.first_divergence = report.details.get("first_divergence")
    cell.min_topk_margin = _min_topk_margin(cell_dir)
    cell.diagnosis_causes = [cause["code"] for cause in diagnosis.causes]
    cell.artifact_path = cell_dir


def _write_phase_artifacts(diagram: PhaseDiagram, *, validate_schema: bool) -> None:
    payload = diagram.to_artifact()
    if validate_schema:
        validate_artifact(payload)
    dump_json(os.path.join(diagram.out_dir, "phase_diagram.json"), payload)
    write_phase_csv(diagram, os.path.join(diagram.out_dir, "phase_diagram.csv"))
    with open(
        os.path.join(diagram.out_dir, "phase_diagram.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(render_phase_text(diagram))


def _min_topk_margin(cell_dir: str) -> float | None:
    trace_dir = os.path.join(cell_dir, "traces")
    if not os.path.isdir(trace_dir):
        return None
    best: float | None = None
    for name in sorted(os.listdir(trace_dir)):
        if not name.endswith(".jsonl"):
            continue
        for row in read_trace(os.path.join(trace_dir, name)):
            for scores in row.get("topk_scores") or []:
                if len(scores) < 2:
                    continue
                margin = abs(float(scores[0]) - float(scores[1]))
                if best is None or margin < best:
                    best = margin
    return best
