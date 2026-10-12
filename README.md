# EmbeddingGemma Code Search Skill

Semantic code search for coding agents (Cursor, Claude Code, Antigravity, …).

> **Design intent (full write-up):** see **[DESIGN.md](./DESIGN.md)** (Tiếng Việt) — architecture, reindex policy, daemon, and rules for contributors.

## Features

- Google **EmbeddingGemma-2** (text/code, Matryoshka dims)
- **Tree-sitter** AST chunking: Python, JS, JSX, TS, TSX, Go, Rust, PHP
- **Shared daemon**: one model in RAM, auto-start, idle unload, short queue
- **Incremental index**: by git commit; dirty/touched files under agent control
- Works as a **Skill** (`SKILL.md` + scripts)

## Quick start

```bash
cd embeddinggemma-code-search
pip install -r scripts/requirements.txt

export CODE_DIR=/path/to/your/project
export CODE_INDEX_DIR=/path/to/your/project/.code_index

python scripts/index.py --force
python scripts/search.py "authentication middleware"
```

## Agent workflow

```text
search  →  discover paths
edit    →  do NOT auto-reindex dirty files
need fresh semantics for edited files?
        →  python scripts/index.py --paths <files>
        →  search again
```

## Layout

```text
embeddinggemma-code-search/
├── DESIGN.md           # ý đồ thiết kế (đọc trước khi phát triển tiếp)
├── SKILL.md            # hướng dẫn agent
├── README.md
└── scripts/
    ├── requirements.txt
    ├── chunker.py
    ├── index.py
    ├── search.py
    ├── daemon.py
    └── client_ipc.py
```

## Official location

Canonical path: **`Thinlt/Grok` → `embeddinggemma-code-search/`**  
(The old standalone `Thinlt/embeddinggemma-code-search` repo was removed.)

## License

Apache-2.0 for integration code. EmbeddingGemma-2 weights are under Apache 2.0 (Google).
