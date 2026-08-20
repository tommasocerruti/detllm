#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

SKIP_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "build",
    "artifacts",
}

TODO_RE = re.compile(r"\bTODO\b[:\- ]?(.*)")
GITHUB_REQUEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class Todo:
    path: str
    line: int
    text: str

    @property
    def todo_id(self) -> str:
        payload = f"{self.path}:{self.line}:{self.text}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()


def _iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def _extract_todos(path: Path) -> Iterable[Todo]:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return []
    todos: list[Todo] = []
    for idx, line in enumerate(content.splitlines(), start=1):
        match = TODO_RE.search(line)
        if not match:
            continue
        text = match.group(1).strip() or "follow up"
        todos.append(Todo(path=str(path), line=idx, text=text))
    return todos


def _github_request(
    method: str,
    url: str,
    token: str,
    payload: dict | None = None,
) -> dict | list:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "detllm-todo-bot",
        },
    )
    with urllib.request.urlopen(req, timeout=GITHUB_REQUEST_TIMEOUT_SECONDS) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _ensure_label(repo: str, token: str, name: str) -> None:
    url = f"https://api.github.com/repos/{repo}/labels"
    try:
        _github_request(
            "POST",
            url,
            token,
            payload={"name": name, "color": "0e8a16", "description": "Auto TODO"},
        )
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 422}:
            raise


def _list_open_issues(repo: str, token: str, label: str) -> list[dict]:
    url = (
        f"https://api.github.com/repos/{repo}/issues"
        f"?state=open&labels={label}&per_page=100"
    )
    data = _github_request("GET", url, token)
    if not isinstance(data, list):
        return []
    return data


def _issue_todo_id(issue: dict) -> str | None:
    body = issue.get("body", "")
    match = re.search(r"<!--\s*todo-id:\s*([a-f0-9]+)\s*-->", body)
    return match.group(1) if match else None


def _create_issue(repo: str, token: str, todo: Todo, label: str) -> None:
    url = f"https://api.github.com/repos/{repo}/issues"
    title = f"TODO: {todo.text}"
    if len(title) > 140:
        title = title[:137] + "..."
    body = (
        f"<!-- todo-id: {todo.todo_id} -->\n"
        f"Found TODO in `{todo.path}:{todo.line}`:\n\n"
        f"> {todo.text}\n\n"
        "Please see `CONTRIBUTING.md` for workflow and conventions.\n"
    )
    _github_request(
        "POST",
        url,
        token,
        payload={"title": title, "body": body, "labels": [label]},
    )


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("Missing GITHUB_REPOSITORY or GITHUB_TOKEN.")
        return 1

    label = os.environ.get("TODO_LABEL", "todo-bot")
    _ensure_label(repo, token, label)

    todos: list[Todo] = []
    for path in _iter_files(Path(".")):
        todos.extend(_extract_todos(path))

    existing = _list_open_issues(repo, token, label)
    existing_ids = {tid for issue in existing if (tid := _issue_todo_id(issue))}

    created = 0
    for todo in todos:
        if todo.todo_id in existing_ids:
            continue
        _create_issue(repo, token, todo, label)
        created += 1

    print(f"Found {len(todos)} TODOs, created {created} issue(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
