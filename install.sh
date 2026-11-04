#!/bin/bash
# install.sh — bootstrap the gemma-code-search skill.
#
#   ./install.sh                                    # venv (uv sync) + shims in ~/.local/bin
#   ./install.sh --skills-global                    # + register for ALL known agents
#   ./install.sh --agent claude,cursor              # + register for the named agents only
#   ./install.sh --project PATH                     # + register inside ONE project
#   ./install.sh --project PATH --agent claude      # combine freely
#
# Shim resolution order at run time:
#   1. EMBEDDINGGEMMA_HOME env override
#   2. nearest .claude|.cursor|.agents/skills/{gemma-code-search,embeddinggemma-code-search} up the tree
#   3. baked-in global skill home (this repo)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$HOME/.local/bin"
# skill name follows the frontmatter in SKILL.md (survives folder renames)
SKILL_NAME="$(grep -m1 '^name:' "$HERE/SKILL.md" | sed 's/^name:[[:space:]]*//' | xargs)"
SKILL_NAME="${SKILL_NAME:-gemma-code-search}"
# pre-rename name: stale registrations under this name are cleaned up on install
LEGACY_SKILL_NAME="${LEGACY_SKILL_NAME:-embeddinggemma-code-search}"

# agent name -> global skills dir (project variants are the same path under the project root)
AGENT_DIRS=(
  "claude:$HOME/.claude/skills"
  "cursor:$HOME/.cursor/skills"
  "antigravity:$HOME/.agents/skills"
  "opencode:$HOME/.config/opencode/skills"
  "gemini:$HOME/.gemini/skills"
)

agent_list() {
  printf '%s' "all, $(printf '%s, ' "${AGENT_DIRS[@]%%:*}" | sed 's/, $//')"
}

usage() {
  cat <<EOF
install.sh — bootstrap the $SKILL_NAME skill

Usage:
  ./install.sh [options]

Options:
  -h, --help            Show this help.
  --skills-global       Register for ALL known agents (machine-wide skill dirs).
  --agent LIST          Comma-separated agents to register (repeatable, combines
                        with --project). Choices: $(agent_list)
  --project PATH        Register inside ONE project (project-local skill dirs
                        instead of machine-wide). PATH may also be positional.
  --skills-dir PATH     Registration target dir override (--project/--agent follow it).

Default (no flags): set up the venv and shims only — no registration.

What each step does:
  1. uv sync      one shared venv at scripts/.venv (uv manages python too).
  2. shims        code-search / code-index into $BIN_DIR; they resolve the
                  skill at run time (EMBEDDINGGEMMA_HOME -> nearest skills dir
                  up the tree -> this copy) and index the git repo you run
                  them from.
  3. registration symlink this skill dir into skills dirs:

                    agent        global dir                 project dir
                    claude       ~/.claude/skills           PATH/.claude/skills
                    cursor       ~/.cursor/skills           PATH/.cursor/skills
                    antigravity  ~/.agents/skills           PATH/.agents/skills
                    opencode     ~/.config/opencode/skills  PATH/.config/opencode/skills
                    gemini       ~/.gemini/skills           PATH/.gemini/skills

Examples:
  ./install.sh                          # venv + shims only
  ./install.sh --skills-global          # venv + shims + every agent, machine-wide
  ./install.sh --agent claude,opencode  # only Claude Code and opencode
  ./install.sh --project ~/work/api     # per-project registration (all agents in that repo)
  ./install.sh --project ~/work/api --agent claude
EOF
}

die() { echo "error: $*" >&2; echo "run './install.sh --help' for usage" >&2; exit 2; }

PROJECT_DIR=""
PROJECT_MODE=0
SKILLS_DIR_OVERRIDE=""
WANT_GLOBAL=0
GLOBAL_AGENTS=()
PROJECT_AGENTS=()

case_agent() { # case_agent <name> — exit on unknown
  [ "$1" = all ] && return 0
  for e in "${AGENT_DIRS[@]}"; do
    [ "${e%%:*}" = "$1" ] && return 0
  done
  die "unknown agent '$1' (choices: $(agent_list))"
}

split_agents() { # split_agents <target:global|project> "a,b,c"
  local target="$1"
  for a in ${2//,/ }; do
    a="$(printf '%s' "$a" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    [ -n "$a" ] || continue
    case_agent "$a"
    if [ "$target" = global ]; then
      GLOBAL_AGENTS+=("$a")
    else
      PROJECT_AGENTS+=("$a")
    fi
  done
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --skills-global) WANT_GLOBAL=1 ;;
    --agent|--agents)
      shift
      if [ $# -eq 0 ]; then die "--agent needs a value"; fi
      split_agents global "$1" ;;
    --agent=*|--agents=*) split_agents global "${1#*=}" ;;
    --project)
      shift
      case "${1:-}" in ""|--*) die "--project needs a PATH" ;; esac
      PROJECT_DIR="$1" ; PROJECT_MODE=1 ;;
    --project=*) PROJECT_DIR="${1#*=}" ; PROJECT_MODE=1 ;;
    --skills-dir)
      shift
      case "${1:-}" in ""|--*) die "--skills-dir needs a PATH" ;; esac
      SKILLS_DIR_OVERRIDE="$1" ;;
    --skills-dir=*) SKILLS_DIR_OVERRIDE="${1#*=}" ;;
    -) die "unknown option '$1'" ;;
    -*) die "unknown option '$1'" ;;
    *)
      if [ -n "$PROJECT_DIR" ]; then die "unexpected extra argument '$1'"; fi
      PROJECT_DIR="$1" ; PROJECT_MODE=1 ;;
  esac
  shift
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
register() { # register <skills-dir>
  mkdir -p "$1"
  # drop a stale registration left by the pre-rename release — only if it still
  # points at a copy of this skill (never remove a user's unrelated dir)
  if [ "$LEGACY_SKILL_NAME" != "$SKILL_NAME" ] && [ -L "$1/$LEGACY_SKILL_NAME" ]; then
    if [ -f "$1/$LEGACY_SKILL_NAME/scripts/search.py" ]; then
      rm -f "$1/$LEGACY_SKILL_NAME"
      echo "[install] removed legacy link $1/$LEGACY_SKILL_NAME"
    fi
  fi
  ln -sfn "$HERE" "$1/$SKILL_NAME"
  echo "[install] linked $1/$SKILL_NAME -> $HERE"
}

dir_for() { # dir_for <agent-name> -> echo global dir
  for e in "${AGENT_DIRS[@]}"; do
    if [ "${e%%:*}" = "$1" ]; then echo "${e#*:}"; return; fi
  done
}

register_list() { # register_list <global|project> [agent...] — "all" expands
  local mode="$1"; shift
  local -a names=()
  for a in "$@"; do
    if [ "$a" = all ]; then
      for e in "${AGENT_DIRS[@]}"; do names+=("${e%%:*}"); done
    else
      names+=("$a")
    fi
  done
  for a in "${names[@]}"; do
    local dir
    if [ -n "$SKILLS_DIR_OVERRIDE" ]; then
      dir="$SKILLS_DIR_OVERRIDE"
    elif [ "$mode" = project ]; then
      dir="$PROJECT_DIR$(dir_for "$a" | sed "s|^$HOME||")"
    else
      dir="$(dir_for "$a")"
    fi
    register "$dir"
  done
}

if [ $PROJECT_MODE -eq 1 ]; then
  if [ ! -e "$PROJECT_DIR" ]; then die "project path '$PROJECT_DIR' does not exist"; fi
  PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
  # --agent values travel with --project: they register inside the project
  for a in ${GLOBAL_AGENTS[@]+"${GLOBAL_AGENTS[@]}"}; do PROJECT_AGENTS+=("$a"); done
  GLOBAL_AGENTS=()
  if [ ${#PROJECT_AGENTS[@]} -eq 0 ]; then PROJECT_AGENTS=(all); fi
fi
if [ $WANT_GLOBAL -eq 1 ] && [ ${#GLOBAL_AGENTS[@]} -eq 0 ]; then
  GLOBAL_AGENTS=(all)
fi

registered=0
if [ ${#GLOBAL_AGENTS[@]} -gt 0 ]; then
  register_list global ${GLOBAL_AGENTS[@]+"${GLOBAL_AGENTS[@]}"}
  registered=1
fi
if [ ${#PROJECT_AGENTS[@]} -gt 0 ]; then
  register_list project ${PROJECT_AGENTS[@]+"${PROJECT_AGENTS[@]}"}
  registered=1
fi
if [ $registered -eq 0 ]; then
  echo "[install] no agents registered (run again with --skills-global, --agent ... or --project PATH)"
fi

echo "[install] done. Try: cd <any-git-repo> && code-search \"rate limiting\" --top-k 4"
