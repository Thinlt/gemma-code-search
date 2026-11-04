#!/usr/bin/env python3
"""
git-ignore filtering for the indexer.

Git's own semantics are used (`git check-ignore`), so nested `.gitignore`
files, `.git/info/exclude`, negations and `**` all behave exactly as they do
for `git status`. Tracked files are never excluded.

A file may live inside a *nested* repository — a directory with its own `.git`
that the outer repository does not track as a submodule. The enclosing
repository's ignore rules still apply to that directory: a plain `ECC` line in
the outer `.gitignore` hides `ECC/` from the outer repository, and the files
inside it should not be indexed. Checking only the nearest repository misses
that rule, because the inner repository cannot ignore its own root, so every
repository from the file's own repo up to the one governing the code dir is
consulted.

No third-party imports here: this module is loaded by the CLI front-ends before
the heavy embedding stack, so `code-search --help` stays fast.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple


def _git(cmd: List[str], cwd: str) -> str:
    try:
        r = subprocess.run(["git"] + cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return ""


def _git_stdin(cmd: List[str], cwd: str, input_text: str) -> str:
    try:
        r = subprocess.run(["git"] + cmd, cwd=cwd, input=input_text, capture_output=True, text=True)
        return r.stdout
    except Exception:
        return ""


def nearest_repo(path: str) -> str:
    """Absolute path of the nearest directory at/above `path` holding a `.git`
    entry, or '' when there is none."""
    cur = Path(path).resolve()
    return next((str(a) for a in (cur, *cur.parents) if (a / ".git").exists()), "")


def _enclosing_repo(repo: str) -> str:
    """Repository enclosing `repo`, or '' when `repo` is top-level."""
    parent = Path(repo).parent
    if parent == Path(repo):
        return ""
    return nearest_repo(str(parent))


def _repo_chain(repo: str, limit: str) -> List[str]:
    """`repo` plus its enclosing repositories, up to and including `limit`.

    Nothing above `limit` is consulted: ignore rules outside the code dir are
    out of scope, so running the tool inside a nested repository still indexes
    that repository even when its parent ignores the whole directory.
    """
    if not repo:
        return []
    chain: List[str] = []
    cur = repo
    while cur:
        chain.append(cur)
        if cur == limit:
            break
        cur = _enclosing_repo(cur)
    return chain


def _path_converters(code_dir: str, repo: str) -> Tuple[Callable[[str], str], Callable[[str], str]]:
    """(to_repo, from_repo) between code-dir-relative and repo-relative paths,
    for a repo at/above (prefix prepended) or below (prefix stripped) the code
    dir. Both sides are resolved first: on macOS a code dir reached through a
    symlink (`/tmp` -> `/private/tmp`) would otherwise look unrelated to its own
    repository and yield a bogus `../..` prefix."""
    code_p = Path(code_dir).resolve()
    repo_p = Path(repo).resolve()
    if repo_p == code_p or repo_p in code_p.parents:
        prefix = os.path.relpath(str(code_p), str(repo_p)).replace(os.sep, "/")
        if prefix == ".":
            return (lambda p: p), (lambda r: r)
        return (lambda p: f"{prefix}/{p}"), (lambda r: r[len(prefix) + 1 :])
    prefix = os.path.relpath(str(repo_p), str(code_p)).replace(os.sep, "/")
    return (
        lambda p: p[len(prefix) + 1 :] if p.startswith(prefix + "/") else p,
        lambda r: f"{prefix}/{r}",
    )


def gitignore_excluded(code_dir: str, rel_paths: Iterable[str]) -> Set[str]:
    """Code-dir-relative paths excluded by git ignore rules.

    Every repository in each file's enclosing chain — from its own repository
    up to the one governing `code_dir` — is consulted, so an outer `.gitignore`
    hiding a nested repository excludes the files inside it. Tracked files are
    never excluded. Returns an empty set when git or a repository is
    unavailable: non-git directories are left untouched by design.
    """
    rels = [p for p in rel_paths if p]
    if not rels:
        return set()

    # everything below works on absolute, symlink-resolved paths so a code dir
    # reached through a link still matches the repository reported by git
    code_p = Path(code_dir).resolve()
    outer = nearest_repo(str(code_p))

    repo_cache: Dict[str, str] = {}

    def repo_for(rel: str) -> str:
        d = str(Path(rel).parent)
        if d not in repo_cache:
            repo_cache[d] = nearest_repo(str(code_p / d))
        return repo_cache[d]

    # repo -> files that must be checked against it
    by_repo: Dict[str, Set[str]] = {}
    for rel in rels:
        repo = repo_for(rel)
        if not repo:
            continue
        for chained in _repo_chain(repo, outer) or [repo]:
            by_repo.setdefault(chained, set()).add(rel)

    excluded: Set[str] = set()
    for repo, under in sorted(by_repo.items()):
        to_repo, from_repo = _path_converters(code_dir, repo)
        tracked = {t for t in _git(["ls-files", "--cached", "-z"], repo).split("\0") if t}
        out = _git_stdin(["check-ignore", "-z", "--stdin"], repo, "\0".join(to_repo(p) for p in sorted(under)))
        for line in out.split("\0"):
            line = line.strip()
            if line and line not in tracked:
                excluded.add(from_repo(line))
    return excluded
