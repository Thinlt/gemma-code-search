---
name: gemma-code-search
description: Fast semantic code search with Google EmbeddingGemma-2, tree-sitter AST chunking (Python, JS, JSX, TS, Go, Rust, PHP), shared model daemon, and commit-based incremental index. Use when finding functions, methods, classes, or logic by meaning — e.g. "where is rate limiting", "find refresh token", "auth middleware".
---

# Gemma Code Search

Semantic search over the codebase with:

- **Tree-sitter AST chunks** (function / class / method)
- Languages: **Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, PHP** (+ markdown fallback)
- **Shared daemon**: model loaded once in RAM; auto-start on first search; idle unload
- **Reindex policy**: on search, only update when **git commit** advances; **dirty/touched files are not auto-reindexed** — you control when to refresh them
- **.gitignore-aware**: files excluded by git ignore rules are skipped. Every repository enclosing a file is consulted, so a directory ignored by a parent repo is excluded even when it is itself a git checkout (nested repo). Ignored entries are pruned on the next rebuild; `--paths` cannot re-add them.
- **Hybrid BM25 + dense**: exact symbol/name matches (camelCase-aware BM25) are fused with semantic ranking (RRF) — better recall when the query names a symbol. Disable per-query with `--no-bm25`

## When to use

- Find code by meaning / behavior (not exact `grep`)
- "Where is the rate limiting logic?"
- "Find refresh token handling"
- "Show AuthService login method"

## Commands

There is a single command: **`code-search`**. Searching is the default action;
index maintenance is the `index` subcommand.

> The installed shim (`code-search`) lives on PATH and runs the skill's
> uv-managed venv — use it from **any project directory**; never call bare
> `python scripts/cli.py`. It targets the containing git repo root
> automatically (`CODE_DIR` is pre-resolved). If the shim is not on PATH, find
> the skill dir (walk-up `.claude|.cursor|.agents/skills/` or
> `EMBEDDINGGEMMA_HOME`) and run `uv run --project <skill>/scripts python
> <skill>/scripts/cli.py ...` (or, without uv,
> `<skill>/scripts/.venv/bin/python <skill>/scripts/cli.py ...`).

```bash
code-search "query"          # search (default)
code-search index [...]      # index maintenance
code-search daemon [...]     # daemon status / stop
code-search --help           # usage for everything
```

### Search (recommended)

```bash
code-search "your natural language query" --top-k 6
```

Exhaustive listing of one topic — every matching chunk, no per-file cap (ranked recall, not a grep substitute):

```bash
code-search "webhook retry" --top-k 100 --max-per-file 0 --json
```

Pure semantic ranking (no BM25 fusion) when you want dense-only behavior:

```bash
code-search "webhook retry" --no-bm25
```

**Encoder consistency**: the index records which encoder built it (`meta.json` → `encoder`). If queries come from a different one (e.g. index built with torch fp32, queries with the ONNX int8 backend), cosine scores are not comparable — search prints a `_encoder mismatch_` warning; rebuild with `code-search index --force` after switching backends.

**Index format versioning**: `meta.json` also records the chunk text format (`text_format`). After a skill update that changes it, search prints `[error] … full rebuild first` (the auto-reindex refuses to run) but still answers from the old index; `--paths`/`--subdir` updates are likewise refused. Run `code-search index --force` **once** to migrate — a plain `code-search index` run also migrates automatically by rebuilding from scratch.

This will:

1. Run **commit-only** incremental reindex if `HEAD` moved — never resumes an interrupted/partial build (it warns `[partial-index]` instead; resume is your explicit call)
2. Auto-start the **daemon** if needed (first time loads the model)
3. Search and print ranked function/class snippets
4. Warn about **dirty files** not yet reindexed

### Continuation hints for documents

Long text documents (`.md`, READMEs, generated docs) are not AST-chunked: the whole file is one source chunk, split into ~320-token parts, with the total line count recorded on every chunk (`total_lines`). A search hit in the middle of such a file comes back as an excerpt that starts at the top and is cut before the matching section. When the source continues past what is shown, the result carries a one-line instruction, e.g.:

```
=> [...] continue reading in <docs/arch.md>: lines 300-1340 (query matched 300-341)
```

The hint sits on its own line, separated from the excerpt by a blank line, and is prefixed `=>` so agents treat it as a retrieval meta-instruction — never as file content. It only states the essentials (the file + the still-unshown line range, plus the matched range when it differs); how to fetch it is the agent's call (Read tool, sed, …). The hint is rendered at search time from the stored line ranges — the embedded text is untouched (`TEXT_FORMAT` stays `3`), so no full rebuild is required; old indexes simply start emitting hints for newly indexed/updated files, and `code-search index --force` migrates everything (cheap: unchanged chunks reuse cached vectors). If you see this marker, trust the line ranges, not the excerpt: read the file at the matched lines before summarizing or editing from the snippet.

### Reindex touched files (agent-controlled)

After you edit files and need the index to match the working tree:

```bash
code-search index --paths src/auth.py src/middleware/rate.py
```

Then search again.

### Other index modes

```bash
# Commit-only incremental (same as auto before search)
code-search index

# Also include uncommitted dirty files
code-search index --working-tree

# Index one subtree into the SHARED index (piece-by-piece, path identity is stable)
code-search index --subdir services/api

# Force re-embed a subtree only, keeping the rest of the shared index
code-search index --subdir services/api --force

# Full rebuild
code-search index --force
```

`--subdir` runs accumulate into one index: search covers everything indexed so far, and already-indexed files (matching file hash) are reused — not re-embedded — by later `--subdir` or full runs.

### Crash-safe resume

Indexing flushes a checkpoint every `EMBEDDINGGEMMA_CKPT_FILES` (default 100) files, each written atomically (tmp + rename) so Ctrl-C can never corrupt the index. If a run is killed (Ctrl-C, crash, reboot), rerun the **same command without `--force`** — it resumes from the last checkpoint instead of starting over.

**Search never auto-resumes**: the auto-reindex before search covers committed changes only; while a build is half-done, search answers from what is indexed and prints a `[partial-index]` warning. Finish the build explicitly with `code-search index`.

**Graceful stop (`q`)**: while indexing in a terminal, press `q` — the in-flight sub-batch is finished, a checkpoint is saved, and the process exits `0` (a stop, **not** a crash/error). Rerun the same command to resume from the checkpoint. Non-TTY runs (pipes, CI, agent shells) are unaffected.

**Progress display**: each file is shown live as it is indexed, compacted to
`<dir1>/<dir2>/...<parent ≤10 chars>/<file ≤32 chars>` — e.g. `services/api/...controller/AuthController.php` (2 root dir levels, collapsed middle levels, immediate parent dir — only when it differs from the level-2 dir — and the file name, tail-kept with a `...` prefix when longer than 32 chars). The most recent file line stays visible above each `[embed]` line (chunking runs far ahead of embedding, so a wiped line would flash for <1s and never be readable). TTY only — non-interactive runs (pipes, CI, agent shells) print the `[embed]`/`[ckpt]` lines only.

**Embedding feedback**: embedding runs in sub-batches of 64 chunks that take tens of seconds with no per-file movement — each one prints `  [embed] N chunks (R reused, M new) ... done Xs (Y ch/s new)` so the silent stretches (especially the first, right after `Model ready on ...`) are visibly working, not hung.

**Embedding cache**: chunk embeddings are content-addressed (sha1 of the embedded text, encoder-matched). Re-indexing an edited file embeds only the chunks whose text actually changed; reruns after `q` and regenerated-but-identical files are free. Vectors from a different encoder (torch fp32 vs ONNX int8) are never reused.

### Daemon ops

```bash
code-search daemon --status   # JSON: pid, model device, idle time, queue depth
code-search daemon --stop     # stop now
code-search daemon --restart  # stop now; the next search starts it again
```

(`code-search daemon` runs inside the same venv as search, so no separate
`<skill-dir>` invocation is needed.)

Idle default: **900s** (15 min). Override with `EMBEDDINGGEMMA_IDLE_SEC`.

## Reliability rules for the agent

1. **Search = discovery**, not source of truth after edits.
2. Index tracks **last committed** code by default.
3. Files you just changed stay **out of auto-reindex** until you run `--paths` or `--working-tree`.
4. Search output may list `dirty (not auto-reindexed)`. If you need those versions, reindex paths then search again.
5. Before editing critical logic from a snippet, **read the live file** at the given path/lines.
6. When a result ends with `[...] continue reading in file <path> ...`, the excerpt was cut mid-document: follow the marked line ranges (especially "the query matched") and read the real file before relying on the snippet.

## Setup

One canonical install of the skill (code + venv + models shared by every
project — the daemon serves all projects from one RAM-resident model; each
project only gets its own `.code_index/`):

```bash
cd gemma-code-search
./install.sh                 # uv sync → scripts/.venv + code-search on PATH
./install.sh --skills-global # additionally register for all harnesses globally
./install.sh --project <repo> # instead register inside one repo (per-project)
```

- **Global** (`--skills-global`): symlinked into `~/.claude/skills`,
  `~/.cursor/skills`, `~/.agents/skills` — available in every project without
  touching the repo. Recommended.
- **Per-project** (`--project <repo>`): symlinked into that repo's
  `.claude/|.cursor/|.agents/skills/` — lets a repo commit/carry the skill
  registration; the venv, models, and daemon stay shared and global.
- The shims resolve the skill at run time: `EMBEDDINGGEMMA_HOME` → nearest
  `.claude|.cursor|.agents/skills/gemma-code-search` up the tree →
  baked-in global home. Copy (not symlink) the skill somewhere and point
  `EMBEDDINGGEMMA_HOME` at it to pin a different copy.

First index (run inside the target repo):

```bash
code-search index --force
code-search "test query"
```

`.code_index/` is appended to the repo `.gitignore` automatically by the shim.

### Optional: int8 ONNX backend (CPU-only machines)

The default backend is **torch** (GPU when available). An int8-quantized ONNX replica of the same pipeline exists for machines without a usable GPU — it uses less RAM and no torch dependency at search time:

```bash
uv sync --extra onnx && uv run --project scripts python scripts/export_onnx.py  # export → quantize → validate vs fp32 (gates: mean cos ≥ 0.97)
EMBEDDINGGEMMA_BACKEND=onnx code-search "query"
```

Measured on M2 Pro: int8 ONNX (CPU EP) is ~2.5x **slower** than torch fp32 on MPS — keep the torch default on GPU machines. The bundle lands in `models/embeddinggemma2-int8/` (gitignored) and activates only via `EMBEDDINGGEMMA_BACKEND=onnx`.

## Environment (optional)

| Variable | Default | Meaning |
|----------|---------|--------|
| `CODE_DIR` | `.` | Repo root to index |
| `CODE_INDEX_DIR` | `./.code_index` | Index storage |
| `TRUNCATE_DIM` | `512` | 768 / 512 / 256 / 128 |
| `EMBEDDINGGEMMA_MAX_SEQ_LEN` | `2048` | Max tokens per chunk fed to the embedder (crash guard) |
| `EMBEDDINGGEMMA_CKPT_FILES` | `100` | Files per checkpoint flush during indexing |
| `EMBEDDINGGEMMA_BATCH_SIZE` | `4` | Embed batch size (measured fastest on M2 Pro under load; try `8`/`16` when the machine is idle) |
| `EMBEDDINGGEMMA_DTYPE` | `auto` | `fp32` / `bf16` / `fp16`; auto = bf16 on CUDA, fp32 on MPS/cpu (measured: bf16 slower via Metal, fp16 gives NaN on Gemma) |
| `EMBEDDINGGEMMA_DEVICE` | `auto` | `cpu` / `mps` / `cuda`; auto = GPU if available, CPU fallback |
| `EMBEDDINGGEMMA_BACKEND` | `torch` | `torch` / `onnx` — onnx needs the exported int8 bundle (see Setup); torch measured faster on GPU machines |
| `EMBEDDINGGEMMA_ONNX_EP` | `auto` | ONNX execution provider; default CPU EP (CoreML partitions this graph heavily — set `coreml` only after measuring) |
| `EMBEDDINGGEMMA_MAX_PER_FILE` | `2` | Max chunks per file in top-k (0 = off); keeps one matching file from flooding results |
| `EMBEDDINGGEMMA_IDLE_SEC` | `900` | Daemon idle unload |
| `EMBEDDINGGEMMA_SOCK` | `/tmp/embeddinggemma.sock` | Unix socket path |
| `EMBEDDINGGEMMA_QUEUE_MAX` | `24` | Max queued searches |
| `EMBEDDINGGEMMA_TCP` | `0` | Set `1` to use TCP (Windows) |

## Install locations

Harness discovery dirs — global (`~/.claude/...`) or per-project (repo-relative); created by `install.sh --skills-global` / `--project`:

| Harness | Path |
|---------|------|
| Cursor | `.cursor/skills/gemma-code-search/` |
| Claude Code | `.claude/skills/gemma-code-search/` |
| Antigravity | `.agents/skills/gemma-code-search/` |

The skill folder may be a symlink to one canonical copy (recommended).
