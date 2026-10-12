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
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from chunker import EXT_TO_LANG, CodeChunk, chunk_file

SUPPORTED_EXTS = set(EXT_TO_LANG.keys()) | {".md"}
# Embedder input cap in tokens. The model's tokenizer reports model_max_length=1e18,
# so without an explicit cap one long chunk can request 100+ GiB tensors.
MAX_SEQ_LEN = int(os.getenv("EMBEDDINGGEMMA_MAX_SEQ_LEN", "2048"))
BATCH_SIZE = 8
CHECKPOINT_FILES = int(os.getenv("EMBEDDINGGEMMA_CKPT_FILES", "100"))
DEVICE_ENV = os.getenv("EMBEDDINGGEMMA_DEVICE", "auto").lower()
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


def resolve_device() -> str:
    if DEVICE_ENV not in ("auto", ""):
        return DEVICE_ENV
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_model(dim: int) -> SentenceTransformer:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    device = resolve_device()
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device=device,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    model.max_seq_length = MAX_SEQ_LEN
    if device != "cpu":
        # gpu self-check: cheap encode; fall back to cpu on error or non-finite output
        try:
            probe = model.encode(
                ["probe"], prompt_name="Document", truncate_dim=dim,
                normalize_embeddings=True, show_progress_bar=False, batch_size=1,
            )
            if not np.isfinite(probe).all():
                raise RuntimeError("non-finite embeddings")
        except Exception as e:
            print(f"[warn] device {device} failed self-check ({e}); using cpu")
            model.to("cpu")
    return model


def embed_chunks(model: SentenceTransformer, chunks: List[CodeChunk], dim: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, dim), dtype=np.float32)
    texts = [c.full_text for c in chunks]
    emb = model.encode(
        texts, prompt_name="Document", truncate_dim=dim,
        normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE,
    )
    return np.array(emb, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="EmbeddingGemma code indexer")
    parser.add_argument("--force", action="store_true", help="Full rebuild")
    parser.add_argument("--working-tree", action="store_true", help="Also reindex dirty/uncommitted files")
    parser.add_argument("--paths", nargs="*", default=None, help="Only reindex these relative paths")
    parser.add_argument(
        "--commit-only",
        action="store_true",
        help="Only reindex files changed since last indexed commit; never resume a backlog "
        "of unindexed files (used by search auto-reindex — resume stays an explicit run)",
    )
    parser.add_argument(
        "--subdir",
        default=None,
        help="Only index files under this directory (relative to --code-dir), into the shared index",
    )
    parser.add_argument("--code-dir", default=os.getenv("CODE_DIR", "."))
    parser.add_argument("--index-dir", default=os.getenv("CODE_INDEX_DIR", "./.code_index"))
    parser.add_argument("--dim", type=int, default=int(os.getenv("TRUNCATE_DIM", "512")))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

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
        if meta.get("dim") != args.dim:
            # embedding width changed; stale vectors cannot be merged — rebuild
            meta, old_docs, old_embs = {}, [], None

    last_commit = meta.get("last_commit", "")
    file_hashes: Dict[str, str] = dict(meta.get("file_hashes", {}))
    current_commit = get_current_commit(code_dir)
    all_files = collect_files(code_dir)
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
                f"(interrupted build or partial scope) — finish with: python scripts/index.py",
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
                        "num_chunks": len(docs),
                        "num_files": len(all_files),
                        "dirty_at_index": dirty_now,
                    },
                    indent=2,
                ).encode("utf-8")
            ),
        )

    groups = [to_process[i : i + CHECKPOINT_FILES] for i in range(0, len(to_process), CHECKPOINT_FILES)]
    model: Optional[SentenceTransformer] = None
    final_docs: List[Dict] = list(kept_docs)
    emb_parts: List[np.ndarray] = list(kept_embs)

    for gi, group in enumerate(groups, 1):
        group_chunks: List[CodeChunk] = []
        for rel in group:
            path = all_files[rel]
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                if len(content) > 120_000:
                    content = content[:120_000]
                group_chunks.extend(chunk_file(rel, content))
                new_hashes[rel] = file_hash(path)
            except Exception as e:
                if not args.quiet:
                    print(f"  skip {rel}: {e}")

        if group_chunks:
            if model is None:
                if not args.quiet:
                    print("Loading EmbeddingGemma-2 for indexing...")
                model = load_model(args.dim)
                if not args.quiet:
                    print(f"Model ready on {model.device}")
            emb_parts.append(np.asarray(embed_chunks(model, group_chunks, args.dim), dtype=np.float32))
            final_docs.extend(c.to_dict() for c in group_chunks)

        persist(final_docs, np.vstack(emb_parts) if emb_parts else np.zeros((0, args.dim), dtype=np.float32))
        if not args.quiet:
            print(f"[ckpt {gi}/{len(groups)}] chunks={len(final_docs)} files_hashed={len(new_hashes)}")

    if not groups:
        persist(final_docs, np.vstack(emb_parts) if emb_parts else np.zeros((0, args.dim), dtype=np.float32))

    if not args.quiet:
        print(f"Done. chunks={len(final_docs)} commit={current_commit[:10]} dirty_files={len(dirty_now)}")
        if dirty_now and not args.working_tree:
            print("Note: dirty (uncommitted) files were NOT reindexed. Agent can run: python scripts/index.py --paths <file> ...")


if __name__ == "__main__":
    main()
