#!/bin/bash
# code-index — EmbeddingGemma indexer shim (same resolution as code-search).
# Usage: code-index [--force|--paths f ...|--working-tree|--subdir d]
set -euo pipefail

resolve_skill() {
  local candidates home d
  if [ -n "${EMBEDDINGGEMMA_HOME:-}" ]; then
    echo "$EMBEDDINGGEMMA_HOME"
    return
  fi
  d="$PWD"
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    for candidates in ".claude" ".cursor" ".agents"; do
      home="$d/$candidates/skills/embeddinggemma-code-search"
      if [ -f "$home/scripts/index.py" ]; then
        echo "$home"
        return
      fi
    done
    d="$(dirname "$d")"
  done
  echo "@SKILL_HOME@"
}

SKILL="$(resolve_skill)"
SCRIPTS="$SKILL/scripts"
[ -d "$SCRIPTS" ] || { echo "error: embeddinggemma-code-search not found (set EMBEDDINGGEMMA_HOME)" >&2; exit 1; }

CODE_DIR="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || pwd)"

if command -v uv >/dev/null 2>&1; then
  exec uv run --project "$SCRIPTS" python "$SCRIPTS/index.py" --code-dir "$CODE_DIR" "$@"
elif [ -x "$SCRIPTS/.venv/bin/python" ]; then
  exec "$SCRIPTS/.venv/bin/python" "$SCRIPTS/index.py" --code-dir "$CODE_DIR" "$@"
else
  echo "error: no uv and no $SCRIPTS/.venv — either 'curl -LsSf https://astral.sh/uv/install.sh | sh'" \
       "or: python3.12 -m venv $SCRIPTS/.venv && $SCRIPTS/.venv/bin/pip install -r $SCRIPTS/requirements.txt" >&2
  exit 1
fi
