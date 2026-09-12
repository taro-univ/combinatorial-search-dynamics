#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  echo "Virtual environment not found. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi

source .venv/bin/activate

if ! command -v codex >/dev/null 2>&1; then
  echo "Codex CLI is not installed. Install it separately from the research environment." >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 \"your prompt\"" >&2
  exit 1
fi

codex run "$*"
