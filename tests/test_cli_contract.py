#!/usr/bin/env python3
"""
Contract tests for the `code-search` front-end.

`scripts/cli.py` dispatches to `search` and `index` by importing them and
calling a known pair of functions:

    search.build_parser() / search.main(argv)
    index.build_parser()  / index.run(args)  / index.main(argv)

Those names are the seam between the command router and the two engines. A
half-applied refactor once left `index.py` on the old `main()`-with-no-arguments
shape while `cli.py` and `search.py` already called `main(argv)`; nothing caught
it, because `index.py` imports the embedding stack and no test imported it.

These tests import the modules and assert the seam exists. They deliberately do
NOT import torch or sentence-transformers: importing `index`/`search` must stay
cheap (that is what keeps `code-search --help` at ~0.05s), and asserting it here
stops the heavy imports from creeping back to module scope.

Run with the skill's venv:
    scripts/.venv/bin/python -m pytest tests/ -q
or without pytest:
    scripts/.venv/bin/python tests/test_cli_contract.py
"""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CLI = SCRIPTS / "cli.py"

sys.path.insert(0, str(SCRIPTS))


def test_cli_is_runnable_without_the_embedding_stack() -> None:
    """`--help` must not pay for torch: it is the first thing anyone runs."""
    probe = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import cli; "
        "print('torch' in sys.modules, 'sentence_transformers' in sys.modules, 'numpy' in sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, check=False, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False False False", proc.stdout


def test_search_exposes_the_parser_and_main_the_router_calls() -> None:
    import search

    assert isinstance(search.build_parser(), argparse.ArgumentParser)
    params = list(inspect.signature(search.main).parameters)
    assert params, "search.main must accept argv (the router passes the sub-arguments)"
    assert params[0] in ("argv", "args")


def test_index_exposes_the_parser_run_and_main_the_router_calls() -> None:
    import index

    assert isinstance(index.build_parser(), argparse.ArgumentParser)
    # the pre-search reindex calls index.run(args) with parsed arguments, not
    # index.main(argv) — both entry points must exist
    assert callable(index.run), "index.run(args) is the in-process entry point search.py uses"
    assert list(inspect.signature(index.run).parameters), "index.run must take the parsed args"
    params = list(inspect.signature(index.main).parameters)
    assert params, "index.main must accept argv (cli.py passes the sub-arguments)"
    assert params[0] in ("argv", "args")


def test_index_run_rejects_a_missing_code_dir_instead_of_crashing() -> None:
    """`run` is also called from search.py, so a bad path must raise SystemExit
    with a message rather than abort the caller with a traceback."""
    import index

    args = index.build_parser().parse_args([])
    args.code_dir = str(ROOT / "does-not-exist-xyz")
    args.index_dir = None
    try:
        index.run(args)
    except SystemExit as e:
        assert "code-dir" in str(e)
    else:  # pragma: no cover - only runs if validation regressed
        raise AssertionError("run() accepted a nonexistent --code-dir")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except (AssertionError, AttributeError, TypeError) as e:
            # a missing attribute or a bad signature IS the regression, not a test bug
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
