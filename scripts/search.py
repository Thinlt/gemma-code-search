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
import subprocess
import sys
from pathlib import Path


def run_index(code_dir: str, index_dir: str, dim: int, quiet: bool = True) -> None:
    """Commit-only incremental reindex (default policy).

    --commit-only keeps search fast: it never resumes a backlog of unindexed
    files (e.g. an interrupted full build) — that stays an explicit index run.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "index.py"),
        "--code-dir",
        code_dir,
        "--index-dir",
        index_dir,
        "--dim",
        str(dim),
        "--commit-only",
    ]
    if quiet:
        cmd.append("--quiet")
    subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="EmbeddingGemma semantic code search")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=None,
        help="Max chunks per file in top-k (0 = unlimited listing); overrides EMBEDDINGGEMMA_MAX_PER_FILE",
    )
    parser.add_argument("--code-dir", default=os.getenv("CODE_DIR", "."))
    parser.add_argument("--index-dir", default=os.getenv("CODE_INDEX_DIR", "./.code_index"))
    parser.add_argument("--dim", type=int, default=int(os.getenv("TRUNCATE_DIM", "512")))
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="Skip auto commit-based reindex before search",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of formatted text",
    )
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Force local in-process model (slow, no shared daemon)",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Disable BM25+dense hybrid fusion (dense cosine only)",
    )
    args = parser.parse_args()

    code_dir = str(Path(args.code_dir).resolve())
    index_dir = str(Path(args.index_dir).resolve())

    if not args.no_reindex:
        run_index(code_dir, index_dir, args.dim, quiet=True)

    if not (Path(index_dir) / "documents.pkl").exists():
        print("Index missing → full build...")
        subprocess.check_call(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "index.py"),
                "--code-dir",
                code_dir,
                "--index-dir",
                index_dir,
                "--dim",
                str(args.dim),
                "--force",
            ]
        )

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
