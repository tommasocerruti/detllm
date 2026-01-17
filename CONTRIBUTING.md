# Contributing to detLLM

Thanks for your interest in detLLM. This project is in early development; please keep changes small and focused.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,dev]'
```

## Running tests

```bash
pytest
```

## Linting (optional)

```bash
pre-commit install
pre-commit run --all-files
```

## Pull request guidelines

- Keep PRs focused on a single goal.
- Include tests for new behavior when feasible.
- Update README or docs when behavior changes.
- Use Conventional Commits for commit messages. See https://www.conventionalcommits.org/en/v1.0.0/
