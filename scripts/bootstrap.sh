#!/usr/bin/env bash
# Bootstrap a development environment (Linux / macOS / Git Bash).
#   ./scripts/bootstrap.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv python install 3.12
uv sync

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

if [ -d .git ]; then
    uv run pre-commit install
else
    echo "Not a git repository; skipping pre-commit install (run 'git init' first)."
fi

uv run python scripts/verify_setup.py
