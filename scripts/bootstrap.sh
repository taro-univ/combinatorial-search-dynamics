#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv." >&2
    exit 1
  fi
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation failed or is not on PATH." >&2
  exit 1
fi

echo "Synchronizing the locked Python 3.11 environment..."
uv sync --all-extras --locked

echo "Setup complete. Next steps:"
echo "  uv run csd doctor"
echo "  uv run csd validate-repository"
