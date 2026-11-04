#!/usr/bin/env python3
"""
Semantic code search client.

- Auto-starts shared daemon (model in RAM) if needed
- Commit-based auto-reindex before search (does not reindex dirty touched files)
- Prints results + dirty-file warnings for the agent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def run_index(code_dir: str, index_dir: str, dim: int, quiet: bool = True) -> None:
    """Commit-only incremental reindex (default policy).

    --commit-only keeps search fast: it never resumes a backlog of unindexed
    files (e.g. an interrupted full build) — that stays an explicit index run.
    """
    import index as index_cli

    args = index_cli.build_parser().parse_args([])
    args.code_dir = code_dir
    args.index_dir = index_dir
    args.dim = dim
    args.commit_only = True
    args.quiet = quiet
    try:
        index_cli.run(args)
    except SystemExit as e:
        # the reindex is best-effort: search still answers from the existing index,
        # so report why it was skipped and carry on
        if e.code not in (None, 0):
            print(e.code, file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Parser for the search command, shared by `code-search` and `search.py`."""
    parser = argparse.ArgumentParser(
        prog="code-search search",
        description=(
            "Search the semantic index for code that matches a natural-language query.\n"
            "Chunks are ranked by hybrid BM25 + embedding similarity (RRF fusion)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  code-search "where is rate limiting applied?"
  code-search "refresh token rotation" --top-k 10
  code-search "webhook retry" --top-k 100 --max-per-file 0 --json
  code-search "auth middleware" --no-bm25     # dense-only ranking

the index is built with:
  code-search index --force
""",
    )
    query_group = parser.add_argument_group("query")
    query_group.add_argument("query", help="what to look for, in natural language")
    query_group.add_argument(
        "--top-k",
        type=int,
        default=6,
        metavar="N",
        help="number of results to return (default: 6)",
    )
    query_group.add_argument(
        "--min-score",
        type=float,
        default=0.12,
        metavar="S",
        help="drop results scoring below this (default: 0.12)",
    )
    query_group.add_argument(
        "--max-per-file",
        type=int,
        default=None,
        metavar="N",
        help="max chunks per file in the result set; 0 = unlimited (default: $EMBEDDINGGEMMA_MAX_PER_FILE or 2)",
    )

    rank_group = parser.add_argument_group("ranking")
    rank_group.add_argument(
        "--no-bm25",
        action="store_true",
        help="disable BM25 + dense fusion and rank on dense cosine similarity alone",
    )

    index_group = parser.add_argument_group("index")
    index_group.add_argument(
        "--no-reindex",
        action="store_true",
        help="skip the commit-based reindex normally run before searching",
    )
    index_group.add_argument(
        "--code-dir",
        default=os.getenv("CODE_DIR", "."),
        metavar="PATH",
        help="repository to search (default: the git repo containing the current directory)",
    )
    index_group.add_argument(
        "--index-dir",
        default=None,
        metavar="PATH",
        help="where the index lives (default: <code-dir>/.code_index)",
    )
    index_group.add_argument(
        "--dim",
        type=int,
        default=int(os.getenv("TRUNCATE_DIM", "512")),
        metavar="N",
        help="embedding dimension: 768, 512, 256 or 128 (default: 512)",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="print raw JSON instead of formatted snippets",
    )
    output_group.add_argument(
        "--no-daemon",
        action="store_true",
        help="run the model in this process instead of the shared daemon (slow; debugging only)",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    code_dir = str(Path(args.code_dir).resolve())
    if not Path(code_dir).is_dir():
        parser.error(
            f"--code-dir must be an existing directory (resolved from cwd), got a file or missing path: {code_dir}"
        )
    # Resolve index_dir relative to code_dir:
    # if code_dir is not '.', default to <code_dir>/.code_index;
    # otherwise keep the old default ./ .code_index
    if args.code_dir.rstrip(os.sep) != "." and args.index_dir is None:
        index_dir = os.path.join(code_dir, ".code_index")
    else:
        index_dir = args.index_dir or "./.code_index"
    index_dir = str(Path(index_dir).resolve())

    if not args.no_reindex:
        run_index(code_dir, index_dir, args.dim, quiet=True)

    if not (Path(index_dir) / "documents.pkl").exists():
        print("Index missing → full build...")
        import index as index_cli

        build_args = index_cli.build_parser().parse_args([])
        build_args.code_dir = code_dir
        build_args.index_dir = index_dir
        build_args.dim = args.dim
        build_args.force = True
        index_cli.run(build_args)

    if args.no_daemon:
        from daemon import do_search, format_results_text

        payload = do_search(
            {
                "query": args.query,
                "index_dir": index_dir,
                "top_k": args.top_k,
                "min_score": args.min_score,
                "dim": args.dim,
                "hybrid": not args.no_bm25,
                **({"max_per_file": args.max_per_file} if args.max_per_file is not None else {}),
            }
        )
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(format_results_text(payload, args.query))
        return

    from client_ipc import send_request

    req = {
        "action": "search",
        "query": args.query,
        "index_dir": index_dir,
        "top_k": args.top_k,
        "min_score": args.min_score,
        "dim": args.dim,
        "hybrid": not args.no_bm25,
        "format": "json" if args.json else "text",
    }
    if args.max_per_file is not None:
        req["max_per_file"] = args.max_per_file
    resp = send_request(req, auto_start=True, verbose=True)

    if args.json:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return

    if resp.get("text"):
        print(resp["text"])
    elif not resp.get("ok"):
        print(f"Search failed: {resp.get('error')} {resp.get('detail', '')}", file=sys.stderr)
        sys.exit(1)
    else:
        from daemon import format_results_text

        print(format_results_text(resp, args.query))


if __name__ == "__main__":
    main()
