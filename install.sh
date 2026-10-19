#!/bin/bash
# install.sh — bootstrap the embeddinggemma-code-search skill.
#
#   ./install.sh                 # venv (uv sync) + shims in ~/.local/bin
#   ./install.sh --skills-global # also register for all harnesses (~/.claude/skills, ~/.cursor/skills, ~/.agents/skills)
#   ./install.sh --project PATH  # register for one project only (project-local symlinks)
#
# Shim resolution order at run time:
#   1. EMBEDDINGGEMMA_HOME env override
#   2. nearest .claude|​.cursor|.agents/skills/embeddinggemma-code-search up the tree (per-project install)
#   3. baked-in global skill home (this repo)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$HOME/.local/bin"
MODE=""

for arg in "$@"; do
  case "$arg" in
    --skills-global) MODE="global" ;;
    --project) MODE="project" ;;
    *) PROJECT_DIR="$arg" ;;
  esac
done

# ---- 1. venv: one uv-managed env inside scripts/ -------------------------
echo "[install] uv sync (wheels cached; first run may download torch)..."
( set -e; cd "$HERE/scripts" && uv sync )

echo "[install] venv ready: $HERE/scripts/.venv"

# ---- 2. shims: substitute the global skill home, install to PATH ---------
mkdir -p "$BIN_DIR"
for shim in code-search code-index; do
  sed "s|@SKILL_HOME@|$HERE|g" "$HERE/shim/$shim.sh" > "$BIN_DIR/$shim"
  chmod +x "$BIN_DIR/$shim"
done
echo "[install] shims: $BIN_DIR/code-search, $BIN_DIR/code-index"

# ---- 3. harness registration --------------------------------------------
HARNESS_DIRS=("$HOME/.claude/skills" "$HOME/.cursor/skills" "$HOME/.agents/skills")

register() { # register <skills-dir>
  mkdir -p "$1"
  ln -sfn "$HERE" "$1/embeddinggemma-code-search"
  echo "[install] linked $1/embeddinggemma-code-search -> $HERE"
}

case "$MODE" in
  global)
    for d in "${HARNESS_DIRS[@]}"; do register "$d"; done
    ;;
  project)
    : "${PROJECT_DIR:?usage: install.sh --project <dir>}"
    PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
    for d in "${HARNESS_DIRS[@]}"; do register "$PROJECT_DIR/$d"; done
    ;;
esac

echo "[install] done. Try: cd <any-git-repo> && code-search \"rate limiting\" --top-k 4"
