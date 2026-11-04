#!/bin/bash
# code-search — EmbeddingGemma semantic search shim.
# Works from any project: resolves the skill installation (project-local first,
# then global), runs it inside its uv-managed venv, and indexes the git repo
# root containing the current directory.
#
#   code-search "query"          search
#   code-search index [...]      build or update the index
#   code-search daemon [...]     inspect or stop the shared model process
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
    for candidates in ".claude" ".cursor" ".agents" ".config/opencode" ".gemini"; do
      # canonical name first, then the pre-rename name (older installs)
      for name in gemma-code-search embeddinggemma-code-search; do
        home="$d/$candidates/skills/$name"
        if [ -f "$home/scripts/cli.py" ]; then
          echo "$home"
          return
        fi
      done
      # renamed copies: any skills/* dir carrying this skill's SKILL.md
      for home in "$d/$candidates"/skills/*/; do
        [ -f "${home}scripts/cli.py" ] || continue
        [ -f "${home}SKILL.md" ] || continue
        # match both the current and the pre-rename skill name
        if grep -Eiq '^name: *(gemma-code-search|.*embeddinggemma)' "${home}SKILL.md"; then
          echo "${home%/}"
          return
        fi
      done
    done
    d="$(dirname "$d")"
  done
  echo "@SKILL_HOME@"
}

SKILL="$(resolve_skill)"
SCRIPTS="$SKILL/scripts"
[ -d "$SCRIPTS" ] || { echo "error: gemma-code-search not found (set EMBEDDINGGEMMA_HOME)" >&2; exit 1; }

# ---- resolve code dir: git repo root, else cwd ---------------------------
# Exported as CODE_DIR, which both subcommands read as their --code-dir default
# (an explicit --code-dir on the command line still wins).
CODE_DIR="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || pwd)"
export CODE_DIR

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
  exec uv run --project "$SCRIPTS" python "$SCRIPTS/cli.py" "$@"
elif [ -x "$SCRIPTS/.venv/bin/python" ]; then
  exec "$SCRIPTS/.venv/bin/python" "$SCRIPTS/cli.py" "$@"
else
  echo "error: no uv and no $SCRIPTS/.venv — either 'curl -LsSf https://astral.sh/uv/install.sh | sh'" \
       "or: python3.12 -m venv $SCRIPTS/.venv && $SCRIPTS/.venv/bin/pip install -r $SCRIPTS/requirements.txt" >&2
  exit 1
fi
