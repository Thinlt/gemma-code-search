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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from chunker import EXT_TO_LANG, CodeChunk, chunk_file

SUPPORTED_EXTS = set(EXT_TO_LANG.keys()) | {".md"}
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


def load_model(dim: int) -> SentenceTransformer:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return SentenceTransformer(
        "google/embeddinggemma-2",
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
    )


def embed_chunks(model: SentenceTransformer, chunks: List[CodeChunk], dim: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, dim), dtype=np.float32)
    texts = [c.full_text for c in chunks]
    emb = model.encode(
        texts, prompt_name="Document", truncate_dim=dim,
        normalize_embeddings=True, show_progress_bar=True, batch_size=32,
    )
    return np.array(emb, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="EmbeddingGemma code indexer")
    parser.add_argument("--force", action="store_true", help="Full rebuild")
    parser.add_argument("--working-tree", action="store_true", help="Also reindex dirty/uncommitted files")
    parser.add_argument("--paths", nargs="*", default=None, help="Only reindex these relative paths")
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
    if meta_path.exists() and docs_path.exists() and emb_path.exists() and not args.force:
        with open(meta_path) as f:
            meta = json.load(f)
        with open(docs_path, "rb") as f:
            old_docs = pickle.load(f)
        old_embs = np.load(emb_path)

    last_commit = meta.get("last_commit", "")
    file_hashes: Dict[str, str] = dict(meta.get("file_hashes", {}))
    current_commit = get_current_commit(code_dir)
    all_files = collect_files(code_dir)

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
        for rel, path in all_files.items():
            h = file_hash(path)
            needs = False
            if rel in commit_changed:
                needs = True
            elif args.working_tree and rel in dirty:
                needs = True
            elif file_hashes.get(rel) != h and rel not in dirty:
                needs = True
            if needs:
                if rel in dirty and not args.working_tree and rel not in commit_changed:
                    continue
                to_process.append(rel)
            else:
                unchanged_paths.add(rel)
        if not args.quiet:
            dirty_note = get_dirty_files(code_dir)
            print(
                f"[incremental] process={len(to_process)} reuse={len(unchanged_paths)} "
                f"dirty={len(dirty_note)} commit={current_commit[:10] if current_commit != 'no-git' else 'n/a'}"
            )

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
    if old_embs is not None and not args.force:
        for i, d in enumerate(old_docs):
            if d["path"] not in reprocess_set and d["path"] in all_files:
                kept_docs.append(d)
                kept_embs.append(old_embs[i])

    new_chunks: List[CodeChunk] = []
    new_hashes = dict(file_hashes)
    for rel in to_process:
        path = all_files[rel]
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if len(content) > 120_000:
                content = content[:120_000]
            chunks = chunk_file(rel, content)
            new_chunks.extend(chunks)
            new_hashes[rel] = file_hash(path)
        except Exception as e:
            if not args.quiet:
                print(f"  skip {rel}: {e}")

    for rel in list(new_hashes):
        if rel not in all_files:
            del new_hashes[rel]

    if not args.quiet:
        print(f"New/updated chunks: {len(new_chunks)}")

    if new_chunks:
        if not args.quiet:
            print("Loading EmbeddingGemma-2 for indexing...")
        model = load_model(args.dim)
        new_embs = embed_chunks(model, new_chunks, args.dim)
        new_docs = [c.to_dict() for c in new_chunks]
    else:
        new_docs = []
        new_embs = np.zeros((0, args.dim), dtype=np.float32)

    final_docs = kept_docs + new_docs
    if kept_embs:
        final_embs = np.vstack([
            np.array(kept_embs, dtype=np.float32),
            np.array(new_embs, dtype=np.float32),
        ])
    else:
        final_embs = np.array(new_embs, dtype=np.float32)

    with open(docs_path, "wb") as f:
        pickle.dump(final_docs, f)
    np.save(emb_path, final_embs)

    dirty_now = sorted(get_dirty_files(code_dir))
    meta = {
        "last_commit": current_commit,
        "file_hashes": new_hashes,
        "dim": args.dim,
        "num_chunks": len(final_docs),
        "num_files": len(all_files),
        "dirty_at_index": dirty_now,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if not args.quiet:
        print(f"Done. chunks={len(final_docs)} commit={current_commit[:10]} dirty_files={len(dirty_now)}")
        if dirty_now and not args.working_tree:
            print("Note: dirty (uncommitted) files were NOT reindexed. Agent can run: python scripts/index.py --paths <file> ...")


if __name__ == "__main__":
    main()
