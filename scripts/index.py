#!/usr/bin/env python3
"""
Incremental indexer for EmbeddingGemma-2.

Policy:
- Auto / default: only reindex files that changed between last_indexed_commit and HEAD
- Does NOT index dirty working-tree changes unless --working-tree or --paths is used
- Agent controls reindex of touched files via --paths
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

import numpy as np

from chunker import EXT_TO_LANG, CodeChunk, chunk_file
from onnx_embedder import encoder_id
from repo_ignore import gitignore_excluded

if TYPE_CHECKING:  # heavy imports stay out of `--help` and of lightweight callers
    from sentence_transformers import SentenceTransformer


def _torch():
    """Import torch on first use. Importing it costs seconds, and the CLI must be
    able to print `--help` (and reach an up-to-date error message) without it."""
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


_TORCH = None

SUPPORTED_EXTS = set(EXT_TO_LANG.keys()) | {".md"}
# Embedder input cap in tokens. The model's tokenizer reports model_max_length=1e18,
# so without an explicit cap one long chunk can request 100+ GiB tensors.
MAX_SEQ_LEN = int(os.getenv("EMBEDDINGGEMMA_MAX_SEQ_LEN", "2048"))
BATCH_SIZE = int(os.getenv("EMBEDDINGGEMMA_BATCH_SIZE", "4"))
CHECKPOINT_FILES = int(os.getenv("EMBEDDINGGEMMA_CKPT_FILES", "100"))
EMBED_SUBBATCH = 64  # chunks per encode call — also the granularity of a graceful 'q' stop
# Version of the embedded string format (what CodeChunk.full_text produces).
# 1 = "title|file|lang|text" (path included), 2 = path dropped, 3 = chunks over
# 320 tokens split into parts titled "(part i/n)" with a signature context
# line on parts 2+. Old vectors embed a different string, so an index from
# another format can only be rebuilt, never merged.
TEXT_FORMAT = 3
DISPLAY_DIR_LEVELS = 2
DISPLAY_DIR_MAX = 10
DISPLAY_FILE_MAX = 32
DEVICE_ENV = os.getenv("EMBEDDINGGEMMA_DEVICE", "auto").lower()
DTYPE_ENV = os.getenv("EMBEDDINGGEMMA_DTYPE", "auto").lower()
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".code_index", "vendor", "target",
    "coverage", ".tox", ".mypy_cache",
}


def run_git(cmd: List[str], cwd: str) -> str:
    try:
        r = subprocess.run(["git"] + cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return ""


def get_current_commit(code_dir: str) -> str:
    return run_git(["rev-parse", "HEAD"], code_dir) or "no-git"


def get_changed_files_between(code_dir: str, old_commit: str, new_commit: str) -> Set[str]:
    if not old_commit or old_commit == "no-git" or not new_commit or new_commit == "no-git":
        return set()
    out = run_git(["diff", "--name-only", old_commit, new_commit], code_dir)
    return {f for f in out.splitlines() if f}


def get_dirty_files(code_dir: str) -> Set[str]:
    dirty: Set[str] = set()
    for args in (["diff", "--name-only"], ["diff", "--name-only", "--cached"]):
        out = run_git(args, code_dir)
        dirty.update(out.splitlines())
    out = run_git(["ls-files", "--others", "--exclude-standard"], code_dir)
    dirty.update(out.splitlines())
    return {f for f in dirty if f}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except Exception:
        return ""
    return h.hexdigest()


def collect_files(code_dir: str) -> Dict[str, Path]:
    root = Path(code_dir)
    files: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            continue
        files[rel] = p
    return files


def subdir_scope(subdir: str, all_files: Dict[str, Path]) -> Set[str]:
    """Paths under `subdir` (relative to code-dir), keeping stable root-relative identity."""
    sub = subdir.replace("\\", "/")
    if sub.startswith("./"):
        sub = sub[2:]
    sub = sub.rstrip("/")
    return {p for p in all_files if p == sub or p.startswith(sub + "/")}


def display_path(rel: str) -> str:
    """Compact one-line progress label: first 2 directory levels, then the
    immediate parent dir (max 10 chars, behind a '...' collapse marker when it
    differs from the level-2 dir), then the file name (max 32 chars, tail-kept
    when truncated). E.g.
    services/api/handlers/v2/controllers/AuthController.php
      → services/api/...controller/AuthController.php
    """
    parts = rel.replace("\\", "/").split("/")
    dirs, fname = parts[:-1], parts[-1]
    shown = dirs[:DISPLAY_DIR_LEVELS]
    if len(dirs) > DISPLAY_DIR_LEVELS:
        parent = dirs[-1]
        if parent != dirs[DISPLAY_DIR_LEVELS - 1]:
            shown.append("..." + parent[:DISPLAY_DIR_MAX])
        else:
            shown.append("...")
    if len(fname) > DISPLAY_FILE_MAX:
        fname = "..." + fname[-(DISPLAY_FILE_MAX - 3):]
    return "/".join(shown + [fname])


_stop_requested = False


def request_stop() -> None:
    global _stop_requested
    _stop_requested = True


def start_stop_listener() -> Optional[Callable[[], None]]:
    """Watch the terminal for a 'q' keypress and request a graceful stop.

    Active only when stdin is a TTY — non-interactive runs (pipes, CI, agent
    shells) are untouched. Returns a restore() that puts the terminal back to
    cooked mode, or None when no listener was started.
    """
    if not sys.stdin.isatty():
        return None
    try:
        import termios
        import tty
    except ImportError:
        termios = None
        try:
            import msvcrt
        except ImportError:
            return None
    else:
        msvcrt = None

    fd = sys.stdin.fileno()
    saved = None
    if termios is not None:
        try:
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            return None

    def _watch() -> None:
        try:
            if msvcrt is not None:
                while not _stop_requested:
                    if msvcrt.kbhit():
                        if msvcrt.getwch().lower() == "q":
                            request_stop()
                            break
                    time.sleep(0.05)
            else:
                while not _stop_requested:
                    ch = sys.stdin.read(1)
                    if not ch:  # EOF (pty closed)
                        break
                    if ch.lower() == "q":
                        request_stop()
                        break
        except Exception:
            pass

    threading.Thread(target=_watch, daemon=True).start()

    def _restore() -> None:
        if saved is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except Exception:
                pass

    atexit.register(_restore)
    return _restore


def resolve_device() -> str:
    if DEVICE_ENV not in ("auto", ""):
        return DEVICE_ENV
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_dtype(device: str):
    """fp32/bf16/fp16 override via EMBEDDINGGEMMA_DTYPE; auto keeps CUDA bf16 but
    stays fp32 on MPS/cpu — measured on M2 Pro: bf16 is ~18% slower through Metal
    and fp16 yields non-finite embeddings (Gemma activation overflow)."""
    d = DTYPE_ENV
    if d in ("fp32", "float32", "float"):
        return _torch().float32
    if d in ("bf16", "bfloat16"):
        return _torch().bfloat16
    if d in ("fp16", "float16", "half"):
        return _torch().float16
    if device == "cuda":
        torch = _torch()
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return _torch().float32


def resolve_backend() -> str:
    """EMBEDDINGGEMMA_BACKEND=auto (default) | anemll | torch | onnx.
    auto = anemll when the Core AI bundle is present and passes its probe,
    else torch. anemll = the ANE package (fails hard if the bundle is absent).
    onnx = int8 CPU bundle: measured on M2 Pro it is slower than torch fp32
    MPS; it earns its keep on CPU-only machines."""
    b = os.getenv("EMBEDDINGGEMMA_BACKEND", "auto").lower()
    if b in ("", "auto"):
        return "auto"
    if b == "torch":
        return "torch"
    if b == "anemll":
        from anemll_embedder import bundle_available

        if not bundle_available():
            raise SystemExit(
                "[error] EMBEDDINGGEMMA_BACKEND=anemll but no bundle — "
                "see scripts/anemll_embedder.py docstring for setup"
            )
        return "anemll"
    if b == "onnx":
        from onnx_embedder import bundle_available

        if not bundle_available():
            raise SystemExit(
                "[error] EMBEDDINGGEMMA_BACKEND=onnx but no bundle — "
                "run: python scripts/export_onnx.py"
            )
        return "onnx"
    if b == "anemll":
        from anemll_embedder import bundle_available

        if not bundle_available():
            raise SystemExit(
                "[error] EMBEDDINGGEMMA_BACKEND=anemll but no bundle — "
                "see scripts/anemll_embedder.py docstring for setup"
            )
        return "anemll"
    return "torch"


def load_model(dim: int) -> "SentenceTransformer":
    backend = resolve_backend()
    if backend in ("auto", "anemll"):
        from anemll_embedder import AnemllEmbedder, bundle_available

        if bundle_available():
            try:
                model = AnemllEmbedder()
                probe = model.encode(
                    ["def add(a, b):\n    return a + b", "x"],
                    prompt_name="Document", truncate_dim=dim,
                    normalize_embeddings=True, batch_size=2,
                )
                # isfinite alone misses the real failure mode seen on macOS 26:
                # Core AI runs the tower but returns all-zero embeddings
                if np.isfinite(probe).all() and float(np.linalg.norm(probe, axis=1).min()) > 0.5:
                    print(f"[model] backend=anemll coreai placement={model.placement}")
                    return model
                print("[warn] anemll tower output is zero (Core AI too old for this package?); falling back to torch")
            except Exception as e:
                print(f"[warn] anemll backend failed ({e}); falling back to torch")

    if backend == "onnx":
        from onnx_embedder import OnnxEmbedder

        try:
            model = OnnxEmbedder()
            probe = model.encode(
                ["probe"], prompt_name="Document", truncate_dim=dim,
                normalize_embeddings=True, batch_size=1,
            )
            if np.isfinite(probe).all():
                print(f"[model] backend=onnx int8 ep={model.provider}")
                return model
            print("[warn] onnx int8 non-finite probe; falling back to torch")
        except Exception as e:
            print(f"[warn] onnx backend failed ({e}); falling back to torch")

    device = resolve_device()

    def build(dev: str, dtype) -> "SentenceTransformer":
        from quiet_logs import silence_import_warnings

        silence_import_warnings()
        from sentence_transformers import SentenceTransformer
        from transformers import logging as hf_logging

        hf_logging.disable_progress_bar()  # drop the "Loading weights" bar — noise in terminals and agent shells
        m = SentenceTransformer(
            "google/embeddinggemma-2",
            device=dev,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        m.max_seq_length = MAX_SEQ_LEN
        return m

    def self_check(m: "SentenceTransformer") -> bool:
        # cheap encode; catches dtype/device failures and non-finite output
        if m.device.type == "cpu":
            return True
        try:
            probe = m.encode(
                ["probe"], prompt_name="Document", truncate_dim=dim,
                normalize_embeddings=True, show_progress_bar=False, batch_size=1,
            )
            return bool(np.isfinite(probe).all())
        except Exception as e:
            print(f"[warn] device {m.device.type} self-check failed ({e})")
            return False

    torch = _torch()
    model = build(device, resolve_dtype(device))
    if not self_check(model):
        model = build(device, torch.float32)
        if not self_check(model):
            print(f"[warn] device {device} failed self-check; using cpu")
            model = build("cpu", torch.float32)
    print(f"[model] device={model.device.type} dtype={next(model.parameters()).dtype}")
    return model


def embed_chunks(model: "SentenceTransformer", chunks: List[CodeChunk], dim: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, dim), dtype=np.float32)
    texts = [c.full_text for c in chunks]
    emb = model.encode(
        texts, prompt_name="Document", truncate_dim=dim,
        normalize_embeddings=True, show_progress_bar=False, batch_size=BATCH_SIZE,
    )
    return np.array(emb, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    """Parser for the index command, reachable as `code-search index`."""
    parser = argparse.ArgumentParser(
        prog="code-search index",
        description=(
            "Build or update the semantic index for a repository.\n"
            "By default only files changed since the last indexed commit are re-embedded;"
            " uncommitted edits stay out until you ask for them with --paths or --working-tree."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  code-search index                       # incremental: re-embed files changed since the last commit
  code-search index --force               # rebuild everything from scratch
  code-search index --working-tree        # also index uncommitted edits
  code-search index --paths src/auth.mjs  # reindex just the files you edited
  code-search index --subdir services/api # index one subtree into the shared index
  code-search index --force --subdir api  # rebuild one subtree, keep the rest

progress and recovery:
  Ctrl-C or 'q' saves a checkpoint; rerun the same command to resume.
  'q' exits 0 — that is a stop, not a failure.

the index is queried with:
  code-search "your query"
""",
    )
    mode_group = parser.add_argument_group("what to index")
    mode_group.add_argument(
        "--force",
        action="store_true",
        help="rebuild from scratch instead of updating incrementally",
    )
    mode_group.add_argument(
        "--working-tree",
        action="store_true",
        help="also index uncommitted (dirty) files",
    )
    mode_group.add_argument(
        "--paths",
        nargs="*",
        default=None,
        metavar="FILE",
        help="reindex only these files, relative to the code dir",
    )
    mode_group.add_argument(
        "--subdir",
        default=None,
        metavar="DIR",
        help="index only files under this directory, into the shared index",
    )
    mode_group.add_argument(
        "--commit-only",
        action="store_true",
        help="limit updates to files changed since the last indexed commit; never resume a "
        "backlog of unindexed files (used by the reindex that runs before a search)",
    )

    location_group = parser.add_argument_group("where")
    location_group.add_argument(
        "--code-dir",
        default=os.getenv("CODE_DIR", "."),
        metavar="PATH",
        help="repository to index (default: the git repo containing the current directory)",
    )
    location_group.add_argument(
        "--index-dir",
        default=None,
        metavar="PATH",
        help="where the index lives (default: <code-dir>/.code_index)",
    )
    location_group.add_argument(
        "--dim",
        type=int,
        default=int(os.getenv("TRUNCATE_DIM", "512")),
        metavar="N",
        help="embedding dimension: 768, 512, 256 or 128 (default: 512)",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress output (used by the reindex before a search)",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute an index run for already-parsed arguments (shared by the CLI and
    the reindex triggered before a search)."""
    code_dir_resolved = Path(args.code_dir).resolve()
    if not code_dir_resolved.is_dir():
        raise SystemExit(
            f"[error] --code-dir must be an existing directory (resolved from cwd), "
            f"got a file or missing path: {code_dir_resolved}"
        )
    # Resolve index-dir relative to code-dir: if code_dir is not '.', index-dir defaults to <code_dir>/.code_index
    code_dir = os.path.abspath(args.code_dir)
    if code_dir != "." and args.index_dir is None:
        args.index_dir = os.path.join(code_dir, ".code_index")
    elif args.index_dir is None:
        args.index_dir = "./.code_index"
    restore_term = start_stop_listener()
    try:
        _run(args)
    finally:
        if restore_term is not None:
            restore_term()


def main(argv: Optional[List[str]] = None) -> None:
    run(build_parser().parse_args(argv))


def _run(args: argparse.Namespace) -> None:
    code_dir = str(Path(args.code_dir).resolve())
    index_dir = Path(args.index_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    docs_path = index_dir / "documents.pkl"
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.json"

    meta: Dict[str, Any] = {}
    old_docs: List[Dict] = []
    old_embs: Optional[np.ndarray] = None
    # --force drops the old index, except when scoped to a subdir (keep the rest)
    if meta_path.exists() and docs_path.exists() and emb_path.exists() and (not args.force or args.subdir):
        with open(meta_path) as f:
            meta = json.load(f)
        with open(docs_path, "rb") as f:
            old_docs = pickle.load(f)
        old_embs = np.load(emb_path)
        if old_embs.shape[0] != len(old_docs):
            # crash between the two index writes; the docs file is authoritative
            old_embs = old_embs[: len(old_docs)]
        incompatible = None
        if meta.get("dim") != args.dim:
            incompatible = f"embedding dim {meta.get('dim')} != {args.dim} (TRUNCATE_DIM changed)"
        elif meta.get("text_format") != TEXT_FORMAT:
            incompatible = (
                f"chunk text format {meta.get('text_format') or 1} != {TEXT_FORMAT} "
                "(embedded strings changed)"
            )
        if incompatible:
            if args.paths is not None or args.subdir or args.commit_only:
                # scoped/incremental updates would merge incompatible vectors into a
                # mixed index (or shrink it to the scope) — only a full rebuild is safe
                sys.exit(
                    f"[error] {incompatible}; cannot update this index in place — "
                    "full rebuild first: code-search index --force"
                )
            # stale vectors cannot be merged with new ones — rebuild from scratch
            print(f"[index-format] {incompatible} — rebuilding from scratch")
            meta, old_docs, old_embs = {}, [], None

    last_commit = meta.get("last_commit", "")
    file_hashes: Dict[str, str] = dict(meta.get("file_hashes", {}))
    current_commit = get_current_commit(code_dir)
    all_files = collect_files(code_dir)
    excluded = gitignore_excluded(code_dir, all_files)
    if excluded:
        for p in excluded:
            all_files.pop(p, None)
        if not args.quiet:
            print(f"[gitignore] skipping {len(excluded)} file(s) matched by .gitignore rules")
    scope: Set[str] = subdir_scope(args.subdir, all_files) if args.subdir else set()
    if args.subdir and not scope:
        if not args.quiet:
            print(f"[subdir] no indexable files under: {args.subdir}")
        return

    to_process: List[str] = []
    unchanged_paths: Set[str] = set()

    if args.paths is not None:
        for p in args.paths:
            p = p.replace("\\", "/").lstrip("./")
            if p in all_files:
                to_process.append(p)
            elif not args.quiet:
                print(f"[warn] path not found or skipped: {p}")
        if not args.quiet:
            print(f"[paths] reindex {len(to_process)} file(s)")
    elif args.force or not old_docs:
        to_process = list(all_files.keys())
        if not args.quiet:
            print(f"[full] indexing {len(to_process)} files...")
    else:
        commit_changed = get_changed_files_between(code_dir, last_commit, current_commit)
        dirty = get_dirty_files(code_dir) if args.working_tree else set()
        backlog = 0
        for rel, path in all_files.items():
            h = file_hash(path)
            needs = False
            if rel in commit_changed:
                needs = True
            elif args.working_tree and rel in dirty:
                needs = True
            elif not args.commit_only and file_hashes.get(rel) != h and rel not in dirty:
                needs = True
            if needs:
                if rel in dirty and not args.working_tree and rel not in commit_changed:
                    continue
                to_process.append(rel)
            else:
                unchanged_paths.add(rel)
                if args.commit_only and rel not in file_hashes:
                    backlog += 1
        if args.commit_only and backlog:
            # an interrupted/partial build must not be resumed by search — surface it instead
            print(
                f"[partial-index] {backlog} file(s) on disk are not in the index "
                f"(interrupted build or partial scope) — finish with: code-search index",
                file=sys.stderr,
            )
        if not args.quiet:
            dirty_note = get_dirty_files(code_dir)
            print(
                f"[incremental] process={len(to_process)} reuse={len(unchanged_paths)} "
                f"dirty={len(dirty_note)} commit={current_commit[:10] if current_commit != 'no-git' else 'n/a'}"
            )

    if args.subdir:
        to_process = [p for p in to_process if p in scope]
        if not args.quiet:
            print(f"[subdir] {args.subdir}: indexing {len(to_process)} file(s) into shared index")

    if not to_process and old_docs and not args.force:
        meta["last_commit"] = current_commit
        meta["dirty_at_index"] = sorted(get_dirty_files(code_dir))
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        if not args.quiet:
            print("Nothing to reindex (commit unchanged / no selected paths).")
        return

    kept_docs: List[Dict] = []
    kept_embs: List[np.ndarray] = []
    reprocess_set = set(to_process)
    if old_embs is not None and (not args.force or args.subdir):
        for i, d in enumerate(old_docs):
            if d["path"] not in reprocess_set and d["path"] in all_files:
                kept_docs.append(d)
                kept_embs.append(old_embs[i])

    new_hashes = dict(file_hashes)
    for rel in list(new_hashes):
        if rel not in all_files:
            del new_hashes[rel]

    dirty_now = sorted(get_dirty_files(code_dir))

    def persist(docs: List[Dict], embs: np.ndarray) -> None:
        # order matters: embeddings, then docs, then hashes — a crash mid-write is
        # reconciled on load; each file lands atomically (tmp + rename) so Ctrl-C
        # can never leave a half-written, unloadable file behind
        def _atomic(path: Path, writer) -> None:
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as f:
                writer(f)
            os.replace(tmp, path)

        _atomic(emb_path, lambda f: np.save(f, embs))
        _atomic(docs_path, lambda f: pickle.dump(docs, f))
        _atomic(
            meta_path,
            lambda f: f.write(
                json.dumps(
                    {
                        "last_commit": current_commit,
                        "file_hashes": new_hashes,
                        "dim": args.dim,
                        "text_format": TEXT_FORMAT,
                        "num_chunks": len(docs),
                        "num_files": len(all_files),
                        "dirty_at_index": dirty_now,
                        # who embedded this index — search warns on mismatch (fp32 vs int8)
                        "encoder": encoder_used or meta.get("encoder", ""),
                    },
                    indent=2,
                ).encode("utf-8")
            ),
        )

    groups = [to_process[i : i + CHECKPOINT_FILES] for i in range(0, len(to_process), CHECKPOINT_FILES)]
    model: Optional[SentenceTransformer] = None
    encoder_used: Optional[str] = None
    final_docs: List[Dict] = list(kept_docs)
    emb_parts: List[np.ndarray] = list(kept_embs)
    pending_hashes: Dict[str, str] = {}  # chunked but not yet embedded; confirmed into new_hashes on flush
    interactive = not args.quiet and sys.stdout.isatty()
    progress_open = False
    done_count = 0

    def close_progress() -> None:
        nonlocal progress_open
        if progress_open:
            # newline keeps the last indexed file visible above each [embed] line —
            # wiping it made file names flash for <1s between embeds (chunking is
            # ~100x faster than embedding), so the ticker looked like it never ran
            print(flush=True)
            progress_open = False

    def show_progress(rel: str) -> None:
        nonlocal progress_open, done_count
        done_count += 1
        if not interactive:
            return
        line = f"  [{done_count}/{len(to_process)}] {display_path(rel)}"
        print(("\r" + line[:118]).ljust(119), end="", flush=True)
        progress_open = True

    # content-addressed embedding cache: sha1(full_text) -> vector. Embedding is a
    # pure function of (text, encoder), so identical chunks — same file edited in
    # one place, re-runs after 'q', regenerated files — reuse the old vector
    # instead of paying GPU time again. Skipped when the index was built by a
    # different encoder (fp32 vs int8 vectors are not interchangeable).
    emb_cache: Dict[str, np.ndarray] = {}

    def prime_cache() -> None:
        if old_embs is None or not old_docs:
            return
        if meta.get("encoder") != encoder_used:
            return
        for d, row in zip(old_docs, old_embs):
            ft = d.get("full_text")
            if ft:
                emb_cache.setdefault(hashlib.sha1(ft.encode()).hexdigest(), row)

    def flush(pending_chunks: List[CodeChunk], pending_rels: List[str]) -> None:
        nonlocal model, encoder_used
        close_progress()
        if pending_chunks:
            if model is None:
                if not args.quiet:
                    print("Loading EmbeddingGemma-2 for indexing...")
                model = load_model(args.dim)
                encoder_used = encoder_id(model, args.dim)
                if not args.quiet:
                    print(f"Model ready on {model.device.type}")
                prime_cache()
            # label the silent stretch: embedding a sub-batch takes tens of seconds
            # with zero output otherwise, which reads as a hang right after "Model ready"
            t0 = time.time()
            rows: List[Optional[np.ndarray]] = []
            todo: List[int] = []
            hashes: List[str] = []
            for i, c in enumerate(pending_chunks):
                h = hashlib.sha1(c.full_text.encode()).hexdigest()
                hashes.append(h)
                v = emb_cache.get(h)
                if v is not None:
                    rows.append(v)
                else:
                    rows.append(None)
                    todo.append(i)
            if todo:
                emb_new = np.asarray(
                    embed_chunks(model, [pending_chunks[i] for i in todo], args.dim),
                    dtype=np.float32,
                )
                for j, i in enumerate(todo):
                    rows[i] = emb_new[j]
                    emb_cache[hashes[i]] = emb_new[j]
            new = len(todo)
            reused = len(pending_chunks) - new
            if not args.quiet:
                extra = f" ({reused} reused, {new} new)" if reused else ""
                print(f"  [embed] {len(pending_chunks)} chunks{extra} ...", end="", flush=True)
            emb_parts.append(np.asarray(rows, dtype=np.float32))
            final_docs.extend(c.to_dict() for c in pending_chunks)
            if not args.quiet:
                dt = time.time() - t0
                rate = f"{new / dt:.1f} ch/s new" if new else "all reused"
                print(f" done {dt:.0f}s ({rate})")
        for rel in pending_rels:
            if rel in pending_hashes:
                new_hashes[rel] = pending_hashes.pop(rel)

    if groups and sys.stdin.isatty() and not args.quiet:
        print(f"Indexing {len(to_process)} file(s) — press 'q' to stop and save progress.")

    stopped = False
    for gi, group in enumerate(groups, 1):
        if _stop_requested:
            stopped = True
            break
        pending_chunks: List[CodeChunk] = []
        pending_rels: List[str] = []
        for rel in group:
            if _stop_requested:
                break
            path = all_files[rel]
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                if len(content) > 120_000:
                    content = content[:120_000]
                pending_chunks.extend(chunk_file(rel, content))
                pending_hashes[rel] = file_hash(path)
                pending_rels.append(rel)
            except Exception as e:
                if not args.quiet:
                    close_progress()
                    print(f"  skip {rel}: {e}")
            show_progress(rel)
            if len(pending_chunks) >= EMBED_SUBBATCH:
                flush(pending_chunks, pending_rels)
                pending_chunks, pending_rels = [], []
                if _stop_requested:
                    break
        if pending_rels:
            # finish and save whatever is in flight — this is also the graceful-stop
            # path: files currently being indexed are saved before we stop
            flush(pending_chunks, pending_rels)
        persist(final_docs, np.vstack(emb_parts) if emb_parts else np.zeros((0, args.dim), dtype=np.float32))
        if not args.quiet:
            close_progress()
            print(f"[ckpt {gi}/{len(groups)}] chunks={len(final_docs)} files_hashed={len(new_hashes)}")
        if _stop_requested:
            stopped = True
            break

    if not groups:
        persist(final_docs, np.vstack(emb_parts) if emb_parts else np.zeros((0, args.dim), dtype=np.float32))

    close_progress()
    if stopped:
        # 'q' pressed: everything completed so far is checkpointed; exit cleanly (not an error)
        print(
            f"[stopped] saved: chunks={len(final_docs)} files_hashed={len(new_hashes)} "
            f"— resume with: code-search index"
        )
        return
    if not args.quiet:
        print(f"Done. chunks={len(final_docs)} commit={current_commit[:10]} dirty_files={len(dirty_now)}")
        if dirty_now and not args.working_tree:
            print("Note: dirty (uncommitted) files were NOT reindexed. Agent can run: code-search index --paths <file> ...")


if __name__ == "__main__":
    main()
