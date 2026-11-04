#!/usr/bin/env python3
"""
code-search — one command for semantic code search and index maintenance.

    code-search "query"           search
    code-search index [...]       build or update the index

Subcommands are dispatched before any heavy import, so `--help` and argument
errors come back in milliseconds even while the embedding stack is unused.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

from repo_ignore import nearest_repo

VERSION = "0.1.0"

# Exit codes, documented in --help so scripts and agents can branch on them.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def _repo_root() -> str:
    """The git repository containing the current directory, else the directory."""
    top = nearest_repo(os.getcwd())
    return top or os.getcwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-search",
        description=(
            "Semantic code search: find functions, methods and classes by what they do,\n"
            "not by what they are called. Indexes tree-sitter AST chunks with Google\n"
            "EmbeddingGemma-2 and ranks them with hybrid BM25 + embedding similarity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="""\
commands:
  search QUERY   find code matching a natural-language description (default)
  index          build or update the index
  daemon         inspect or stop the shared model process

run 'code-search <command> --help' for the options of a command.

examples:
  code-search "where is rate limiting applied?"
  code-search "refresh token rotation" --top-k 10 --json
  code-search index --paths src/auth.py
  code-search daemon --status

how it works:
  The index lives in <project>/.code_index. Source is chunked along AST
  boundaries (Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, PHP, plus
  markdown), so a hit is usually a single function or class. Files excluded by
  git ignore rules — including a directory hidden by an enclosing repository's
  .gitignore — are never indexed.

  A search reindexes files changed since the last indexed commit, then asks a
  shared daemon (one model in RAM, idle-unloading) to embed the query and rank
  the chunks. Uncommitted edits are *not* picked up automatically: reindex the
  files you edited with 'code-search index --paths <file>' when you need them.

environment:
  EMBEDDINGGEMMA_HOME        skill installation directory
  EMBEDDINGGEMMA_BACKEND     torch (default), onnx, anemll
  EMBEDDINGGEMMA_IDLE_SEC    daemon idle unload, default 900
  TRUNCATE_DIM               embedding dimension, default 512
  (see 'code-search search --help' and 'code-search index --help' for the rest)

exit status:
  0  success (including a stopped index run saved with 'q')
  1  runtime error (no index, daemon failure, invalid repository)
  2  usage error (unknown option or command)
""",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="show this help message and exit",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="show the version and exit",
    )
    return parser


COMMANDS: Dict[str, str] = {
    "search": "find code matching a natural-language description",
    "index": "build or update the index",
    "daemon": "inspect or stop the shared model process",
}


def print_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help()
    print()


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Root-level flags and command routing happen before the heavy imports.
    if not argv:
        print_help(build_parser())
        return EXIT_OK
    if argv[0] in ("-h", "--help"):
        print_help(build_parser())
        return EXIT_OK

    command = argv[0]
    rest = argv[1:]
    if command in ("-V", "--version"):
        print(f"code-search {VERSION}")
        return EXIT_OK

    if command in COMMANDS:
        # `code-search index --foo` is far more often a mistyped index run than a
        # search whose query is literally "index", so the subcommand wins. When
        # the remaining arguments are exactly a search's, the mistake is cheap to
        # make and hard to see, so name the fix explicitly.
        if command in ("index", "daemon") and any(a.startswith("--") for a in rest):
            search_only = {"--top-k", "--min-score", "--max-per-file", "--no-reindex", "--no-bm25",
                           "--no-daemon", "--json"}
            if search_only & set(rest):
                raise SystemExit(
                    f"[error] {command!r} is a command, not part of a query.\n"
                    f'to search for that word:  code-search search "{command}" '
                    + " ".join(rest) + "\n"
                    f"to {COMMANDS[command]}:  code-search {command} --help"
                )
        return _dispatch(command, rest)

    # Anything else is a query: `code-search "rate limiting"` and
    # `code-search rate limiting` both read naturally.
    return _run_search(argv)


def _dispatch(command: str, rest: List[str]) -> int:
    if command == "search":
        return _run_search(rest)
    if command == "index":
        return _run_index(rest)
    return _run_daemon(rest)


def _run_search(argv: List[str]) -> int:
    from search import main as search_main

    search_main(argv)
    return EXIT_OK


def _run_index(argv: List[str]) -> int:
    import index as index_cli

    index_cli.main(argv)
    return EXIT_OK


def _run_daemon(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="code-search daemon",
        description=(
            "Inspect or stop the shared model daemon that serves every project.\n"
            "It starts on the first search and unloads its model after an idle period."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  code-search daemon --status   # JSON: pid, device, uptime, idle time, queue depth
  code-search daemon --stop     # ask the daemon to exit now
""",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="print the daemon's current state as JSON")
    action.add_argument("--stop", action="store_true", help="stop the daemon")
    action.add_argument("--restart", action="store_true", help="stop the daemon, then start it on the next search")
    args = parser.parse_args(argv)

    if not (args.status or args.stop or args.restart):
        parser.print_help()
        return EXIT_OK

    from client_ipc import send_request

    if args.restart:
        try:
            send_request({"action": "shutdown"}, auto_start=False)
        except Exception:
            pass
        print("Daemon stopped; it will start again on the next search.")
        return EXIT_OK

    action_name = "stats" if args.status else "shutdown"
    try:
        print_json(send_request({"action": action_name}, auto_start=False))
    except Exception as e:
        if args.status:
            print_json({"ok": False, "running": False, "error": str(e)})
        else:
            print_json({"ok": False, "error": str(e)})
            return EXIT_ERROR
    return EXIT_OK


def print_json(payload: object) -> None:
    import json

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
