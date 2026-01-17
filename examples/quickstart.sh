#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,hf]'

detllm env --out artifacts/env.json

detllm run --backend hf --model distilgpt2 --prompt "Hello" --tier 1 --batch-size 1 --out artifacts/run1

detllm check --backend hf --model distilgpt2 --prompt "Hello" --tier 1 --runs 3 --batch-size 1 --out artifacts/check1

detllm check --backend hf --model distilgpt2 --prompt "Hello" --tier 1 --runs 3 --batch-size 1 --vary-batch 1,2 --out artifacts/check2
