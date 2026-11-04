# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Open-source project scaffolding: Apache-2.0 `LICENSE`, `NOTICE`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and pull request templates, and CI.

### Changed

- `README.md` is now a standalone project README (previously scoped to the monorepo).
- `DESIGN.md` §8 records this repository as the canonical source.

## [0.1.0]

Initial public snapshot of the skill, extracted from the `Thinlt/Grok` monorepo with its
development history intact.

### Added

- Semantic code search over tree-sitter AST chunks (function / method / class) with
  Google **EmbeddingGemma-2** embeddings.
- Language support: Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, PHP, plus a markdown
  fallback for long prose documents.
- Hybrid ranking: identifier-aware BM25 fused with dense cosine similarity via RRF
  (`--no-bm25` for dense-only).
- Shared daemon holding one model in RAM: auto-start on first search, FIFO request queue,
  idle unload after `EMBEDDINGGEMMA_IDLE_SEC`.
- Incremental index keyed on git commits, with `--paths` / `--working-tree` for
  agent-controlled reindexing of edited files.
- `.gitignore`-aware file walking (nested ignore files, negations, `**`, tracked files exempt).
- Crash-safe indexing: atomic checkpoints every `EMBEDDINGGEMMA_CKPT_FILES` files, graceful
  stop with `q`, resume by rerunning without `--force`.
- Content-addressed embedding cache, so only genuinely changed chunks are re-embedded.
- Continuation hints (`=> continue reading in <file>: lines …`) for truncated excerpts of
  long documents.
- Optional int8 **ONNX** backend (`scripts/export_onnx.py`) and optional Apple **ANE** backend,
  both falling back to torch when their artifacts are unavailable.
- `install.sh` with per-agent registration (Claude Code, Cursor, Antigravity, opencode,
  Gemini CLI) and rename-tolerant skill discovery, plus `code-search` / `code-index` shims.
- `SKILL.md`, the agent-facing skill contract.

[Unreleased]: https://github.com/Thinlt/gemma-code-search/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Thinlt/gemma-code-search/releases/tag/v0.1.0
