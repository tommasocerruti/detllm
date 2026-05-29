"""Project initialization helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from detllm.usability.config import config_path_in, prompts_path_in, starter_config

PROMPTS_JSONL = """\
{"prompt": "Choose one: A or B. Answer with a single letter."}
{"prompt": "Complete the phrase with one word: deterministic inference is"}
"""


@dataclass(frozen=True)
class InitResult:
    status: str
    config_path: str
    prompt_path: str
    next_commands: list[str]


def init_project(
    out_dir: str = ".",
    *,
    profile: str = "local-debug",
    force: bool = False,
) -> InitResult:
    if profile != "local-debug":
        raise ValueError("Only the local-debug init profile is supported")
    os.makedirs(out_dir, exist_ok=True)
    config_path = config_path_in(out_dir)
    prompt_path = prompts_path_in(out_dir)
    existing = [str(path) for path in [config_path, prompt_path] if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Generated files already exists; refusing to overwrite: {', '.join(existing)}"
        )

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(starter_config(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    prompt_path.write_text(PROMPTS_JSONL, encoding="utf-8")
    return InitResult(
        status="PASS",
        config_path=str(config_path),
        prompt_path=str(prompt_path),
        next_commands=[
            "detllm doctor",
            "detllm profile list",
            "detllm profile run quick_check",
        ],
    )
