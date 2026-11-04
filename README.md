# Gemma Code Search

Semantic code search for coding agents — powered by Google **EmbeddingGemma-2**, **tree-sitter** AST
chunking, and a shared model daemon.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Ask for behavior, not for strings:

```bash
code-search "where is rate limiting applied?"
code-search "refresh token rotation"
code-search "AuthService login method"
```

Works as a **skill** for Claude Code, Cursor, Antigravity, opencode and Gemini CLI, and as a plain
CLI from any project.

---

## Why

`grep` finds strings; LSP finds symbols. Neither finds *meaning*. Gemma Code Search complements
them: it returns the smallest semantic unit — a function, method or class — ranked by a hybrid of
BM25 (identifier-aware, camelCase-split) and dense embedding similarity fused with RRF.

- **Google EmbeddingGemma-2** — text/code embeddings with Matryoshka dimensions (768 / 512 / 256 / 128).
- **Tree-sitter AST chunking** — Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, PHP (+ markdown fallback).
- **Shared daemon** — one model in RAM for every project; auto-starts on first search, unloads when idle.
- **Commit-based incremental index** — updates when `git HEAD` advances; files you are editing stay
  untouched until *you* say so.
- **.gitignore-aware** — nested `.gitignore`, `.git/info/exclude`, negations and `**` are honored.

Read [DESIGN.md](DESIGN.md) (Tiếng Việt) for the full design rationale.

## Requirements

- Python **3.12** (managed automatically by [uv](https://docs.astral.sh/uv/))
- `git`
- macOS (Apple Silicon — MPS, optional ANE backend), Linux (CUDA or CPU), or Windows (TCP transport)

The first run downloads the EmbeddingGemma-2 weights from Hugging Face (a few GB) and caches them.

## Install

```bash
git clone https://github.com/Thinlt/gemma-code-search.git
cd gemma-code-search

./install.sh                  # uv sync + the `code-search` command on PATH
./install.sh --skills-global  # + register the skill for every known agent (recommended)
./install.sh --project <repo> # …or register inside a single repository
```

See `./install.sh --help` for per-agent registration (`--agent claude,cursor,opencode,…`).

Then, in any git repository:

```bash
code-search index --force   # first full build
code-search "authentication middleware"
```

## Usage

There is one command, `code-search`. Searching is the default action; index
maintenance lives in the `index` subcommand, and the shared daemon in `daemon`.

```bash
code-search "query"          # search
code-search index [...]      # build or update the index
code-search daemon [...]     # daemon status / stop
code-search --help           # full usage
```

### Search

```bash
code-search "your natural language query" --top-k 6
```

| Flag | Meaning |
|------|---------|
| `--top-k N` | Number of results (default `6`) |
| `--min-score F` | Minimum similarity score (default `0.12`) |
| `--max-per-file N` | Cap chunks per file (`0` = off) |
| `--no-bm25` | Dense-only ranking (skip hybrid fusion) |
| `--no-reindex` | Skip the automatic commit-based reindex before searching |
| `--dim N` | Matryoshka dimension: `768` / `512` / `256` / `128` |
| `--json` | Machine-readable output |

Exhaustive listing of one topic — every matching chunk, no per-file cap:

```bash
code-search "webhook retry" --top-k 100 --max-per-file 0 --json
```

### Index

```bash
code-search index                     # commit-only incremental (same as the auto step before search)
code-search index --force             # full rebuild
code-search index --paths src/auth.py # reindex specific files (the agent-controlled path)
code-search index --working-tree      # include uncommitted dirty files
code-search index --subdir services/api    # index one subtree into the shared index
```

### Agent workflow

```text
search  →  discover paths
edit    →  dirty files are NOT auto-reindexed
need fresh semantics for edited files?
        →  code-search index --paths <files>
        →  search again
```

Long text documents (`.md`, READMEs) return an excerpt plus a `=> continue reading in <file>: lines …`
hint. Trust the line ranges — read the real file before acting on the snippet.

### Daemon

```bash
code-search daemon --status
code-search daemon --stop
```

The daemon idle-unloads after `EMBEDDINGGEMMA_IDLE_SEC` (default `900`). Every environment
variable is documented in [SKILL.md](SKILL.md#environment-optional).

## Repository layout

```text
.
├── DESIGN.md             # design intent (Tiếng Việt)
├── SKILL.md              # agent-facing skill instructions
├── README.md
├── install.sh            # venv + shims + agent registration
├── shim/
│   └── code-search.sh    # PATH shim, resolves the skill at run time
└── scripts/
    ├── cli.py            # `code-search` entry point (search / index / daemon)
    ├── chunker.py        # tree-sitter AST chunking
    ├── repo_ignore.py    # git-ignore filtering (nested-repo aware)
    ├── quiet_logs.py     # mutes harmless import-time warnings (torch/ST)
    ├── index.py          # index CLI implementation
    ├── search.py         # search CLI implementation
    ├── daemon.py         # shared model daemon + BM25/dense fusion
    ├── client_ipc.py     # Unix socket / TCP client
    ├── onnx_embedder.py  # optional int8 ONNX backend
    ├── export_onnx.py    # export + quantize + validate the ONNX bundle
    ├── anemll_embedder.py   # optional Apple ANE backend
    ├── anemll_worker.py
    ├── pyproject.toml
    └── requirements.txt
```

## Troubleshooting

**`error: gemma-code-search not found`** — set `EMBEDDINGGEMMA_HOME` to the clone directory,
or re-run `./install.sh`.

**`_encoder mismatch_` warning** — the index was built with a different backend (torch vs ONNX int8).
Rebuild with `code-search index --force`.

**`[error] … full rebuild first`** — the chunk text format changed after an update. Run
`code-search index --force` once to migrate.

**`[partial-index]` warning** — a previous build was interrupted. Run `code-search index` again
(without `--force`) to resume from the last checkpoint.

**Slow first search** — the daemon is loading the model into RAM. Later searches reuse it.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). Please read [DESIGN.md](DESIGN.md) before changing architecture.

## Security

See [SECURITY.md](SECURITY.md) for how to report a vulnerability.

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attributions.

The EmbeddingGemma-2 model weights are published by Google under Apache 2.0; they are downloaded at
install time and are not distributed with this repository.
