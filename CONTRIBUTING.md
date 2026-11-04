# Contributing to Gemma Code Search

Thanks for your interest in improving Gemma Code Search. This document explains how to set up a
development environment, the conventions we follow, and how to get a change merged.

## Code of Conduct

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report
unacceptable behavior as described there.

## Ways to contribute

- **Bug reports** — open an issue with the bug template. Include your OS, Python version, hardware
  (Apple Silicon / CUDA / CPU) and the exact command that failed.
- **Feature requests** — open an issue with the feature template. Describe the problem first, then
  the proposed solution.
- **Documentation** — typos, unclear sections and better examples are all welcome.
- **Code** — bug fixes, new language grammars, new backends, performance work.

> **Before architectural changes**, read [DESIGN.md](DESIGN.md). It records the decisions that are
> already settled (daemon + IPC, commit-only reindex, agent-controlled touched files, skill-first
> packaging). A pull request that reverses one of those decisions needs an issue discussing it first.

## Development setup

Requirements: Python **3.12**, [`uv`](https://docs.astral.sh/uv/), and `git`.

```bash
git clone https://github.com/Thinlt/gemma-code-search.git
cd gemma-code-search

uv sync --project scripts                 # create the venv
uv sync --project scripts --extra onnx    # …plus the optional ONNX backend
```

Run the CLIs straight from the checkout:

```bash
uv run --project scripts python scripts/index.py  --code-dir <repo> --force
uv run --project scripts python scripts/search.py --code-dir <repo> "rate limiting"
```

Installing the shims (`./install.sh`) is optional for development and will point them at your
checkout.

## Project layout

| Path | Responsibility |
|------|----------------|
| `scripts/chunker.py` | Tree-sitter parsing and AST-aware chunk splitting |
| `scripts/index.py` | Indexing CLI, checkpointing, incremental/hash bookkeeping |
| `scripts/search.py` | Search CLI |
| `scripts/daemon.py` | Shared model process, request queue, BM25 + dense fusion |
| `scripts/client_ipc.py` | Unix-socket / TCP client used by both CLIs |
| `scripts/onnx_embedder.py`, `scripts/export_onnx.py` | Optional int8 ONNX backend and its exporter |
| `scripts/anemll_embedder.py`, `scripts/anemll_worker.py` | Optional Apple ANE backend |
| `shim/*.sh` | Thin PATH wrappers that resolve the skill at run time |
| `install.sh` | venv creation, shim install, per-agent registration |

`SKILL.md` is the agent-facing contract and is loaded verbatim by coding agents — keep it accurate
and terse. `DESIGN.md` is the human-facing design record and is written in Tiếng Việt.

## Conventions

- **Language** — code, comments, docstrings, commit messages and `SKILL.md` in English.
  `DESIGN.md` stays in Tiếng Việt.
- **Style** — [Ruff](https://docs.astral.sh/ruff/) is the reference linter, line length 120. Match the
  surrounding code and keep type hints on public functions. Note that the existing scripts are **not**
  yet `ruff format`-clean: do not reformat files wholesale in an unrelated pull request — keep the
  diff focused, and run `ruff format` only on files you are already changing substantially.
- **Dependencies** — add a runtime dependency to **both** `scripts/pyproject.toml` and
  `scripts/requirements.txt`, then refresh the lockfile with `uv lock --project scripts`.
- **No new always-on services.** Anything long-running belongs in the daemon and must idle-unload.
- **Backends are optional.** A new encoder backend must degrade gracefully — if its artifacts are
  missing, fall back to torch rather than crash (see the existing ONNX and ANE embedders).

## Tests and checks

There is no unit-test suite yet; verification is currently manual and end-to-end. Before opening a
pull request, run:

```bash
uvx ruff check scripts     # must be clean (config in scripts/pyproject.toml)
```

and exercise the change against a real repository:

```bash
uv run --project scripts python scripts/index.py  --code-dir /path/to/repo --force
uv run --project scripts python scripts/search.py --code-dir /path/to/repo "your query" --top-k 6
```

If your change touches chunking or the index format, also verify that an existing index still loads
or that the migration path (`code-index --force`) prints the expected message. Please state in the
pull request which OS and backend (torch/MPS/CUDA/ONNX/ANE) you tested on.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
`refactor:`, `perf:`, `chore:`. Keep the subject in the imperative mood and under ~72 characters.

```text
feat(chunker): split PHP traits like classes
fix(index): keep checkpoint atomic on interrupt
docs(readme): document --min-score
```

## Pull request process

1. Fork the repository and branch from `main` (`feat/short-description`).
2. Make a focused change — one logical change per pull request.
3. Run the checks above and describe what you tested.
4. Fill in the pull request template.
5. Link the issue your change addresses, if any.
6. A maintainer will review; please respond to review comments rather than force-pushing over them.

By submitting a contribution you agree that it is licensed under the [Apache License 2.0](LICENSE),
as described in section 5 of that license.
