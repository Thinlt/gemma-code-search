---
name: embeddinggemma-code-search
description: Fast semantic code search with Google EmbeddingGemma-2, tree-sitter AST chunking (Python, JS, JSX, TS, Go, Rust, PHP), shared model daemon, and commit-based incremental index. Use when finding functions, methods, classes, or logic by meaning — e.g. "where is rate limiting", "find refresh token", "auth middleware".
---

# EmbeddingGemma Code Search

Semantic search over the codebase with:

- **Tree-sitter AST chunks** (function / class / method)
- Languages: **Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, PHP** (+ markdown fallback)
- **Shared daemon**: model loaded once in RAM; auto-start on first search; idle unload
- **Reindex policy**: on search, only update when **git commit** advances; **dirty/touched files are not auto-reindexed** — you control when to refresh them

## When to use

- Find code by meaning / behavior (not exact `grep`)
- "Where is the rate limiting logic?"
- "Find refresh token handling"
- "Show AuthService login method"

## Commands

### Search (recommended)

```bash
python scripts/search.py "your natural language query" --top-k 6
```

This will:

1. Run **commit-only** incremental reindex if `HEAD` moved
2. Auto-start the **daemon** if needed (first time loads the model)
3. Search and print ranked function/class snippets
4. Warn about **dirty files** not yet reindexed

### Reindex touched files (agent-controlled)

After you edit files and need the index to match the working tree:

```bash
python scripts/index.py --paths src/auth.py src/middleware/rate.py
```

Then search again.

### Other index modes

```bash
# Commit-only incremental (same as auto before search)
python scripts/index.py

# Also include uncommitted dirty files
python scripts/index.py --working-tree

# Full rebuild
python scripts/index.py --force
```

### Daemon ops

```bash
python scripts/daemon.py --status
python scripts/daemon.py --stop
```

Idle default: **900s** (15 min). Override with `EMBEDDINGGEMMA_IDLE_SEC`.

## Reliability rules for the agent

1. **Search = discovery**, not source of truth after edits.
2. Index tracks **last committed** code by default.
3. Files you just changed stay **out of auto-reindex** until you run `--paths` or `--working-tree`.
4. Search output may list `dirty (not auto-reindexed)`. If you need those versions, reindex paths then search again.
5. Before editing critical logic from a snippet, **read the live file** at the given path/lines.

## Setup

```bash
pip install -r scripts/requirements.txt
python scripts/index.py --force    # first index
python scripts/search.py "test query"
```

Add `.code_index/` to `.gitignore`.

## Environment (optional)

| Variable | Default | Meaning |
|----------|---------|--------|
| `CODE_DIR` | `.` | Repo root to index |
| `CODE_INDEX_DIR` | `./.code_index` | Index storage |
| `TRUNCATE_DIM` | `512` | 768 / 512 / 256 / 128 |
| `EMBEDDINGGEMMA_IDLE_SEC` | `900` | Daemon idle unload |
| `EMBEDDINGGEMMA_SOCK` | `/tmp/embeddinggemma.sock` | Unix socket path |
| `EMBEDDINGGEMMA_QUEUE_MAX` | `24` | Max queued searches |
| `EMBEDDINGGEMMA_TCP` | `0` | Set `1` to use TCP (Windows) |

## Install locations

| Harness | Path |
|---------|------|
| Cursor | `.cursor/skills/embeddinggemma-code-search/` |
| Claude Code | `.claude/skills/embeddinggemma-code-search/` |
| Antigravity | `.agents/skills/embeddinggemma-code-search/` |
