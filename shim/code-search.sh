#!/bin/bash
# code-search — EmbeddingGemma semantic search shim.
# Works from any project: resolves the skill installation (project-local first,
# then global), runs it inside its uv-managed venv, and indexes the git repo
# root containing the current directory.
set -euo pipefail

# ---- resolve skill dir ---------------------------------------------------
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
      if [ -f "$home/scripts/search.py" ]; then
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

# ---- resolve code dir: git repo root, else cwd ---------------------------
CODE_DIR="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || pwd)"

# ---- keep each project's .code_index out of git (once) -------------------
if [ -t 0 ] || [ "${EMBEDDINGGEMMA_EDIT_GITIGNORE:-1}" = "1" ]; then
  if [ -d "$CODE_DIR/.git" ] && [ -w "$CODE_DIR" ]; then
    touch "$CODE_DIR/.gitignore"
    if ! grep -qs '^\.code_index/$' "$CODE_DIR/.gitignore"; then
      printf '\n.code_index/\n' >> "$CODE_DIR/.gitignore"
    fi
  fi
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run --project "$SCRIPTS" python "$SCRIPTS/search.py" --code-dir "$CODE_DIR" "$@"
elif [ -x "$SCRIPTS/.venv/bin/python" ]; then
  exec "$SCRIPTS/.venv/bin/python" "$SCRIPTS/search.py" --code-dir "$CODE_DIR" "$@"
else
  echo "error: no uv and no $SCRIPTS/.venv — either 'curl -LsSf https://astral.sh/uv/install.sh | sh'" \
       "or: python3.12 -m venv $SCRIPTS/.venv && $SCRIPTS/.venv/bin/pip install -r $SCRIPTS/requirements.txt" >&2
  exit 1
fi
