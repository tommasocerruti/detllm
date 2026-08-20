"""Render Flight Recorder diagnosis artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detllm.flight_recorder.diagnose import Diagnosis


def render_diagnosis(diagnosis: Diagnosis, *, include_token_text: bool = False) -> str:
    lines = [
        f"Status: {diagnosis.status}",
        f"Category: {diagnosis.category}",
        f"Summary: {diagnosis.summary}",
        "",
        "Likely causes:",
    ]
    for index, cause in enumerate(diagnosis.causes, start=1):
        lines.append(f"{index}. {cause['title']} ({cause['code']}, {cause['confidence']})")
        lines.append(f"   Evidence: {cause['explanation']}")
        lines.append(f"   Next experiment: {cause['next_step']}")

    if diagnosis.probe_plan:
        lines.extend(["", "Probe plan:"])
        for item in diagnosis.probe_plan:
            lines.append(f"- {item['probe']}: {item['reason']}")

    if include_token_text:
        lines.extend(["", "Token text: requested when available in artifacts."])

    return "\n".join(lines) + "\n"
