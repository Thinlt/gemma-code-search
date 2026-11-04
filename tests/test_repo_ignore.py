#!/usr/bin/env python3
"""
Regression tests for git-ignore filtering.

The case that motivated these tests: a directory listed in an outer
repository's `.gitignore` that is itself a git checkout (a nested repository,
not a submodule). The inner repository cannot ignore its own root, so asking
only the nearest repository indexed the whole directory — an `ECC/` folder
hidden by the parent `.gitignore` was fully indexed.

Run with the skill's venv:
    scripts/.venv/bin/python -m pytest tests/ -q
or without pytest:
    scripts/.venv/bin/python tests/test_repo_ignore.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from repo_ignore import gitignore_excluded

GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env={**GIT_ENV})


def make_repo(path: Path, files: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", ".")
    for name, content in files.items():
        f = path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    git(path, "add", "-A")
    git(path, "commit", "-qm", "init")


def nested_ignored_fixture(root: Path) -> Path:
    """outer/ (git) ignores `vendor-lib/`, which is its own git repo."""
    outer = root / "outer"
    make_repo(outer, {".gitignore": "vendor-lib\n", "src/app.py": "def app():\n    pass\n"})
    make_repo(outer / "vendor-lib", {"secret.py": "def leak():\n    pass\n"})
    return outer


def test_nested_ignored_repo_is_excluded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        outer = nested_ignored_fixture(Path(tmp))
        rels = ["src/app.py", "vendor-lib/secret.py"]
        excluded = gitignore_excluded(str(outer), rels)
        assert "vendor-lib/secret.py" in excluded, "nested repo hidden by the outer .gitignore must be excluded"
        assert "src/app.py" not in excluded


def test_running_inside_the_nested_repo_still_indexes_it() -> None:
    """Ignore rules above the code dir are out of scope: entering a nested repo
    directly must index it even when its parent ignores the directory."""
    with tempfile.TemporaryDirectory() as tmp:
        outer = nested_ignored_fixture(Path(tmp))
        nested = outer / "vendor-lib"
        excluded = gitignore_excluded(str(nested), ["secret.py"])
        assert excluded == set()


def test_tracked_file_is_never_excluded() -> None:
    """A file that git tracks stays indexable even if a pattern matches it."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        # commit the log first, then add the ignore rule: a tracked file is not
        # retroactively excluded (`git add -f` equivalent)
        make_repo(repo, {"keep.log": "x\n"})
        (repo / ".gitignore").write_text("*.log\n")
        (repo / "untracked.log").write_text("y\n")
        git(repo, "add", ".gitignore")
        git(repo, "commit", "-qm", "ignore logs")
        excluded = gitignore_excluded(str(repo), ["keep.log", "untracked.log"])
        assert excluded == {"untracked.log"}, excluded


def test_non_git_directory_is_untouched() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "plain"
        plain.mkdir()
        (plain / ".gitignore").write_text("skipped.py\n")
        (plain / "skipped.py").write_text("x = 1\n")
        assert gitignore_excluded(str(plain), ["skipped.py"]) == set()


def test_symlinked_code_dir_still_matches_its_repo() -> None:
    """macOS reaches /tmp through /private/tmp; a code dir given via the symlink
    must still line up with the repository git reports."""
    with tempfile.TemporaryDirectory() as tmp:
        outer = nested_ignored_fixture(Path(tmp))
        link = Path(tmp) / "link"
        try:
            link.symlink_to(outer, target_is_directory=True)
        except OSError:
            return  # platform without symlink support
        excluded = gitignore_excluded(str(link), ["vendor-lib/secret.py", "src/app.py"])
        assert "vendor-lib/secret.py" in excluded


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
