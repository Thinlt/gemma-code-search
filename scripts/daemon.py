#!/usr/bin/env python3
"""
Shared EmbeddingGemma daemon.

- Loads model once into RAM
- Serves search over Unix socket (or TCP fallback)
- Auto idle shutdown
- Short request queue for concurrent agents
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import re
import signal
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx_embedder import encoder_id

SOCKET_PATH = os.getenv("EMBEDDINGGEMMA_SOCK", "/tmp/embeddinggemma.sock")
PID_FILE = os.getenv("EMBEDDINGGEMMA_PID", "/tmp/embeddinggemma.daemon.pid")
LOCK_FILE = os.getenv("EMBEDDINGGEMMA_LOCK", "/tmp/embeddinggemma.daemon.lock")
IDLE_SEC = int(os.getenv("EMBEDDINGGEMMA_IDLE_SEC", "900"))
QUEUE_MAX = int(os.getenv("EMBEDDINGGEMMA_QUEUE_MAX", "24"))
INDEX_CACHE_MAX = int(os.getenv("EMBEDDINGGEMMA_INDEX_CACHE", "3"))
MAX_PER_FILE = int(os.getenv("EMBEDDINGGEMMA_MAX_PER_FILE", "2"))
# continuation hint: how much stored content is shown per result, what the
# hint marker looks like ("=> " prefix + marker so agents treat the line as
# meta-instruction, never as file content), and how many lines a "read more"
# suggestion covers
RESULT_CONTENT_CAP = 2500
CONT_MARK = "[...] "
CONT_HINT_LABEL = "=> "
CONT_JUMP = 160
TRUNCATE_DIM = int(os.getenv("TRUNCATE_DIM", "512"))
HOST = os.getenv("EMBEDDINGGEMMA_HOST", "127.0.0.1")
PORT = int(os.getenv("EMBEDDINGGEMMA_PORT", "8765"))
USE_TCP = os.getenv("EMBEDDINGGEMMA_TCP", "0") == "1" or sys.platform == "win32"

_model = None
_last_active = time.time()
_shutting_down = False
_queue: Optional[asyncio.Queue] = None
_index_cache: "OrderedDict[str, Tuple[List[Dict], np.ndarray, Dict, Tuple[int, int]]]" = OrderedDict()
_dim = TRUNCATE_DIM


def log(msg: str) -> None:
    print(f"[daemon] {msg}", flush=True)


def touch_activity() -> None:
    global _last_active
    _last_active = time.time()


def load_model():
    global _model, _dim
    if _model is not None:
        return _model
    from quiet_logs import silence_import_warnings

    silence_import_warnings()
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import logging as hf_logging

    hf_logging.disable_progress_bar()  # drop the "Loading weights" bar — noise in logs

    log("Loading EmbeddingGemma-2 (text-only)...")

    backend = os.getenv("EMBEDDINGGEMMA_BACKEND", "auto").lower()
    if backend in ("auto", "anemll"):
        from anemll_embedder import AnemllEmbedder, bundle_available

        if bundle_available():
            try:
                m = AnemllEmbedder()
                probe = m.encode(
                    ["def add(a, b):\n    return a + b", "x"],
                    prompt_name="Document", truncate_dim=_dim,
                    normalize_embeddings=True, batch_size=2,
                )
                # isfinite alone misses the real failure mode seen on macOS 26:
                # Core AI runs the tower but returns all-zero embeddings
                if np.isfinite(probe).all() and float(np.linalg.norm(probe, axis=1).min()) > 0.5:
                    _model = m
                    log(f"Model ready on anemll coreai placement={m.placement}")
                    return _model
                log("[warn] anemll tower output is zero (Core AI too old for this package?); using torch")
            except Exception as e:
                log(f"[warn] anemll backend failed ({e}); using torch")
        elif backend == "anemll":
            log("[warn] EMBEDDINGGEMMA_BACKEND=anemll but no bundle; using torch (see anemll_embedder.py docstring)")

    if backend == "onnx":
        from onnx_embedder import OnnxEmbedder, bundle_available

        if bundle_available():
            try:
                m = OnnxEmbedder()
                probe = m.encode(
                    ["probe"], prompt_name="CodeRetrieval", truncate_dim=_dim,
                    normalize_embeddings=True, batch_size=1,
                )
                if np.isfinite(probe).all():
                    _model = m
                    log(f"Model ready on onnx int8 ep={m.provider}")
                    return _model
                log("[warn] onnx int8 non-finite probe; using torch")
            except Exception as e:
                log(f"[warn] onnx backend failed ({e}); using torch")
        else:
            log("[warn] EMBEDDINGGEMMA_BACKEND=onnx but no bundle; using torch (run scripts/export_onnx.py)")

    def build(dtype) -> "SentenceTransformer":
        return SentenceTransformer(
            "google/embeddinggemma-2",
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"vision_config": None, "audio_config": None},
        )

    def resolve_dtype(device_type: str):
        d = os.getenv("EMBEDDINGGEMMA_DTYPE", "auto").lower()
        if d in ("fp32", "float32", "float"):
            return torch.float32
        if d in ("bf16", "bfloat16"):
            return torch.bfloat16
        if d in ("fp16", "float16", "half"):
            return torch.float16
        # measured on M2 Pro/MPS: bf16 ~18% slower through Metal, fp16 non-finite
        if device_type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return torch.float32

    # sentence-transformers auto-selects cuda > mps > cpu
    try:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    except Exception:
        device_type = "cpu"

    dtype = resolve_dtype(device_type)
    _model = build(dtype)
    if _model.device.type != "cpu":
        try:
            probe = _model.encode(
                ["probe"], prompt_name="CodeRetrieval", truncate_dim=_dim,
                normalize_embeddings=True, show_progress_bar=False, batch_size=1,
            )
            if not np.isfinite(probe).all():
                raise RuntimeError("non-finite embeddings")
        except Exception as e:
            log(f"[warn] dtype {dtype} failed self-check ({e}); retrying fp32")
            dtype = torch.float32
            _model = build(dtype)
    log(f"Model ready on {_model.device} dtype={dtype}")
    return _model


def get_index(index_dir: str) -> Tuple[List[Dict], np.ndarray, Dict]:
    index_dir = str(Path(index_dir).resolve())
    docs_path = Path(index_dir) / "documents.pkl"
    emb_path = Path(index_dir) / "embeddings.npy"
    meta_path = Path(index_dir) / "meta.json"

    if not docs_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Index missing under {index_dir}")

    # signature of the data files: a checkpoint flush while the daemon is alive
    # must swap the cached snapshot without needing a restart
    sig = (docs_path.stat().st_mtime_ns, emb_path.stat().st_mtime_ns)
    cached = _index_cache.get(index_dir)
    if cached is not None and cached[3] == sig:
        _index_cache.move_to_end(index_dir)
        return cached[0], cached[1], cached[2]

    with open(docs_path, "rb") as f:
        documents = pickle.load(f)
    embeddings = np.load(emb_path)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    _index_cache[index_dir] = (documents, embeddings, meta, sig)
    while len(_index_cache) > INDEX_CACHE_MAX:
        _index_cache.popitem(last=False)
    return documents, embeddings, meta


def invalidate_index_cache(index_dir: Optional[str] = None) -> None:
    if index_dir is None:
        _index_cache.clear()
    else:
        _index_cache.pop(str(Path(index_dir).resolve()), None)


RRF_K = 60  # reciprocal rank fusion constant (standard k=60)
_bm25_cache: "OrderedDict[str, Tuple[Any, Tuple[int]]]" = OrderedDict()

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+")


def code_tokens(text: str) -> List[str]:
    """Identifier-aware BM25 tokens: split camelCase/PascalCase boundaries and keep
    the raw lowercased form too, so 'refreshToken' matches the query 'refresh token'."""
    out: List[str] = []
    for w in re.findall(r"[A-Za-z0-9]+", text or ""):
        out.append(w.lower())
        parts = [p.lower() for p in _CAMEL.findall(w) if p]
        if len(parts) > 1:
            out.extend(parts)
    return out


def get_bm25(index_dir: str, documents: List[Dict]):
    """BM25 over chunk path+name+content; rebuilt when documents.pkl changes."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None
    key = str(Path(index_dir).resolve())
    try:
        sig = (Path(index_dir, "documents.pkl").stat().st_mtime_ns,)
    except OSError:
        return None
    cached = _bm25_cache.get(key)
    if cached is not None and cached[1] == sig:
        _bm25_cache.move_to_end(key)
        return cached[0]
    corpus = [
        code_tokens(f"{d.get('path', '')} {d.get('name', '')} {d.get('content', '')}") or ["_empty_"]
        for d in documents
    ]
    bm25 = BM25Okapi(corpus)
    _bm25_cache[key] = (bm25, sig)
    while len(_bm25_cache) > INDEX_CACHE_MAX:
        _bm25_cache.popitem(last=False)
    return bm25


def _part_group(doc: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[str], str]]:
    """Identity of the source chunk a split part belongs to (chunker
    TEXT_FORMAT 3): parts share path/kind/parent/name. None = whole chunk."""
    if doc.get("part") is None:
        return None
    return (doc.get("path") or "", doc.get("kind") or "", doc.get("parent"), doc.get("name") or "")


def _continuation_hint(
    path: str,
    kind: str,
    shown_from: int,
    shown_through: int,
    total_end: int,
    matched: Optional[Tuple[Optional[int], Optional[int]]],
) -> str:
    """One compact line appended to an excerpt whose source continues beyond
    what is shown — a long document (md/README) whose whole-file chunk was
    split by the chunker, or stitched content cut at the preview cap. Only
    states the essentials — the file and the line range still to read, plus
    the matched range when it differs — leaving how to fetch it to the agent.
    """
    m0, m1 = matched if matched else (None, None)
    # anchor the range at the matched section when it lies past the excerpt
    jump_from = int(m0) if m0 and int(m0) > shown_through else shown_through + 1
    jump_to = total_end if total_end >= jump_from else jump_from + CONT_JUMP - 1
    hint = f"{CONT_HINT_LABEL}{CONT_MARK}continue reading in <{path}>: lines {jump_from}-{jump_to}"
    m1 = int(m1) if m1 else (int(m0) if m0 else None)
    whole_span = bool(m0) and int(m0) <= shown_from and m1 >= jump_to
    if m0 and m1 > shown_through and not whole_span and (int(m0), m1) != (jump_from, jump_to):
        hint += f" (query matched {int(m0)}-{m1})"
    return hint


def _build_result(
    documents: List[Dict],
    idx: int,
    has_parts: bool,
    part_groups: Optional[Dict[Tuple[str, str, Optional[str], str], List[int]]],
    score: float,
) -> Dict[str, Any]:
    """One search result: parts stitched back into the source chunk, excerpt
    capped, with a continuation hint appended when the source keeps going."""
    doc = documents[int(idx)]
    content = doc.get("content", "")
    start_line, end_line = doc.get("start_line"), doc.get("end_line")
    matched = (start_line, end_line)  # range of the part that matched, before stitching
    if has_parts and doc.get("part") is not None:
        # stitch the parts back into the whole source chunk: tail matches
        # return the full method/class, not just the matching slice
        members = sorted(
            (part_groups or {}).get(_part_group(doc), [idx]),
            key=lambda i: documents[i]["part"][0],
        )
        content = "".join(documents[i].get("content", "") for i in members)
        start_line = min(documents[i].get("start_line", 0) for i in members)
        end_line = max(documents[i].get("end_line", 0) for i in members)
    # where does the shown excerpt end, and does the source continue past it?
    # (long documents exceed the char cap; document tail parts are trimmed to
    # fit the token budget). total_lines is recorded by the chunker for
    # document chunks; old indexes fall back to the stitched end line.
    excerpt = content[:RESULT_CONTENT_CAP]
    total_end = int(doc.get("total_lines") or 0) or int(end_line or 0)
    shown_through = int(start_line or 1) + excerpt.count("\n")
    truncated = len(content) > RESULT_CONTENT_CAP
    # end metadata beyond the excerpt only proves continuation for document
    # (file-level / total_lines-carrying) chunks — for code chunks the stored
    # end line can exceed the stripped content's own lines without any
    # content missing, so truncation is the only trigger there
    continues = truncated or (
        (doc.get("kind") == "file" or doc.get("total_lines")) and total_end > shown_through
    )
    continuation = ""
    if continues:
        continuation = _continuation_hint(
            path=doc.get("path") or "",
            kind=doc.get("kind") or "",
            shown_from=int(start_line or 1),
            shown_through=shown_through,
            total_end=total_end,
            matched=matched,
        )
    return {
        "path": doc.get("path"),
        "name": doc.get("name"),
        "kind": doc.get("kind"),
        "parent": doc.get("parent"),
        "start_line": start_line,
        "end_line": end_line,
        "language": doc.get("language"),
        "score": round(score, 4),
        "content": excerpt + (f"\n\n{continuation}" if continuation else ""),
        "continuation": continuation or None,
    }


def do_search(req: Dict[str, Any]) -> Dict[str, Any]:
    model = load_model()
    index_dir = req.get("index_dir") or "./.code_index"
    top_k = int(req.get("top_k", 6))
    min_score = float(req.get("min_score", 0.12))
    query = req.get("query") or ""
    dim = int(req.get("dim", _dim))

    if not query.strip():
        return {"ok": False, "error": "empty_query"}

    try:
        documents, embeddings, meta = get_index(index_dir)
    except FileNotFoundError as e:
        return {"ok": False, "error": "index_missing", "detail": str(e)}

    q_emb = model.encode(
        query, prompt_name="CodeRetrieval", truncate_dim=dim, normalize_embeddings=True
    )
    scores = np.dot(embeddings, q_emb)
    top_idx = np.argsort(scores)[::-1]

    max_per_file = int(req.get("max_per_file", MAX_PER_FILE))
    scan_limit = max(top_k * 20, 100)
    pool_size = max(50, top_k * 8)

    # dense candidates: above min_score, in cosine order
    dense_pool: List[int] = []
    for idx in top_idx[:scan_limit]:
        if scores[int(idx)] < min_score:
            break  # scores are descending: no later candidate can pass
        dense_pool.append(int(idx))
        if len(dense_pool) >= pool_size:
            break

    # bm25 candidates: exact-token matches on identifiers; hybrid recall for
    # symbol names the dense encoder dilutes (e.g. 'RetryPolicy', 'AuthService')
    bm25_pool: List[int] = []
    if req.get("hybrid", True):
        bm25 = get_bm25(index_dir, documents)
        if bm25 is not None:
            q_toks = code_tokens(query)
            if q_toks:
                b_scores = bm25.get_scores(q_toks)
                order = np.argsort(b_scores)[::-1][:pool_size]
                bm25_pool = [int(i) for i in order if b_scores[i] > 0]

    if bm25_pool:
        dense_rank = {idx: r for r, idx in enumerate(dense_pool)}
        bm25_rank = {idx: r for r, idx in enumerate(bm25_pool)}

        def rrf(i: int) -> float:
            s = 0.0
            if i in dense_rank:
                s += 1.0 / (RRF_K + dense_rank[i])
            if i in bm25_rank:
                s += 1.0 / (RRF_K + bm25_rank[i])
            return s

        fused = sorted(set(dense_pool) | set(bm25_pool), key=rrf, reverse=True)
    else:
        fused = dense_pool

    # small-to-big: split parts of one source chunk compete as a single entry —
    # fused is score-ordered, so each group's first occurrence is its best part.
    # Without this, the k slots fill with sibling parts of one long method.
    part_groups: Dict[Tuple[str, str, Optional[str], str], List[int]] = {}
    has_parts = any(d.get("part") is not None for d in documents)
    if has_parts:
        for i, d in enumerate(documents):
            g = _part_group(d)
            if g is not None:
                part_groups.setdefault(g, []).append(i)
        seen: set = set()
        collapsed: List[int] = []
        for idx in fused:
            g = _part_group(documents[int(idx)])
            if g is not None:
                if g in seen:
                    continue
                seen.add(g)
            collapsed.append(int(idx))
        fused = collapsed

    # diversity: without a per-file cap, one file whose many chunks all match
    # (e.g. an API client for exactly this query) floods the whole top-k and
    # pushes every other relevant file out of view
    per_file: Dict[str, int] = {}
    picked: List[int] = []
    for idx in fused:
        path = documents[int(idx)].get("path") or ""
        if max_per_file > 0 and per_file.get(path, 0) >= max_per_file:
            continue
        per_file[path] = per_file.get(path, 0) + 1
        picked.append(int(idx))
        if len(picked) >= top_k:
            break

    results = [
        _build_result(documents, idx, has_parts, part_groups, float(scores[idx]))
        for idx in picked
    ]

    payload = {
        "ok": True,
        "results": results,
        "indexed_commit": meta.get("last_commit", ""),
        "dirty_at_index": meta.get("dirty_at_index", []),
        "num_chunks": meta.get("num_chunks", len(documents)),
        "hybrid_used": bool(bm25_pool),
    }
    # an index embedded by one encoder must not be searched by another: cosine
    # scores across fp32-vs-int8 encoders are not comparable
    cur_enc = encoder_id(model, dim)
    idx_enc = meta.get("encoder") or f"embeddinggemma-2:torch:float32:{dim}"  # pre-onnx indexes
    if cur_enc != idx_enc:
        payload["encoder_mismatch"] = {"index": idx_enc, "current": cur_enc}
    return payload


def format_results_text(payload: Dict[str, Any], query: str) -> str:
    if not payload.get("ok"):
        return f"ERROR: {payload.get('error')} {payload.get('detail', '')}"

    lines = [f'### Semantic search: "{query}"\n']
    commit = payload.get("indexed_commit") or "n/a"
    lines.append(f"_indexed_commit: `{commit[:12] if commit != 'n/a' else commit}`_\n")

    dirty = payload.get("dirty_at_index") or []
    if dirty:
        preview = ", ".join(dirty[:8])
        more = f" (+{len(dirty) - 8} more)" if len(dirty) > 8 else ""
        lines.append(
            f"_dirty (not auto-reindexed): {preview}{more}_\n"
            f"_To refresh those files: `code-search index --paths <file>`_\n"
        )

    mismatch = payload.get("encoder_mismatch")
    if mismatch:
        lines.append(
            f"_encoder mismatch: index built with `{mismatch['index']}`, queries use "
            f"`{mismatch['current']}` — rebuild with `code-search index --force` "
            f"for consistent scores_\n"
        )

    results = payload.get("results") or []
    if not results:
        lines.append("No results above min_score.")
        return "\n".join(lines)

    for rank, r in enumerate(results, 1):
        title = f"{r.get('kind')}: {r.get('name')}"
        if r.get("parent"):
            title = f"{r.get('kind')}: {r['parent']}.{r.get('name')}"
        loc = f"{r.get('path')}:{r.get('start_line')}-{r.get('end_line')}"
        lang = r.get("language") or ""
        lines.append(f"**[{rank}] {title}**  ({loc})  [{lang}]  score={r.get('score')}")
        fence = lang if lang in ("python", "javascript", "typescript", "go", "rust", "php") else ""
        lines.append(f"```{fence}")
        lines.append(r.get("content") or "")
        lines.append("```\n")
    return "\n".join(lines)


async def handle_request(raw: str) -> str:
    global _shutting_down
    touch_activity()

    try:
        req = json.loads(raw)
    except json.JSONDecodeError:
        return json.dumps({"ok": False, "error": "invalid_json"})

    action = req.get("action", "search")

    if action == "ping":
        return json.dumps({
            "ok": True,
            "pid": os.getpid(),
            "idle_remaining": max(0, IDLE_SEC - (time.time() - _last_active)),
            "queue_size": _queue.qsize() if _queue else 0,
            "model_loaded": _model is not None,
        })

    if action == "shutdown":
        _shutting_down = True
        return json.dumps({"ok": True, "message": "shutting_down"})

    if action == "invalidate":
        invalidate_index_cache(req.get("index_dir"))
        return json.dumps({"ok": True})

    if action == "stats":
        return json.dumps({
            "ok": True,
            "pid": os.getpid(),
            "idle_sec": IDLE_SEC,
            "idle_remaining": max(0, IDLE_SEC - (time.time() - _last_active)),
            "queue_size": _queue.qsize() if _queue else 0,
            "queue_max": QUEUE_MAX,
            "index_cache": list(_index_cache.keys()),
            "model_loaded": _model is not None,
        })

    if action != "search":
        return json.dumps({"ok": False, "error": "unknown_action"})

    if _shutting_down:
        return json.dumps({"ok": False, "error": "shutting_down"})

    assert _queue is not None
    if _queue.qsize() >= QUEUE_MAX:
        return json.dumps({"ok": False, "error": "queue_full"})

    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    await _queue.put((req, fut))
    try:
        result = await asyncio.wait_for(fut, timeout=120)
    except asyncio.TimeoutError:
        return json.dumps({"ok": False, "error": "timeout"})

    if req.get("format") == "text":
        text = format_results_text(result, req.get("query", ""))
        return json.dumps({"ok": result.get("ok", False), "text": text, **{k: v for k, v in result.items() if k != "results"}})
    return json.dumps(result)


async def worker() -> None:
    assert _queue is not None
    while not _shutting_down:
        try:
            req, fut = await asyncio.wait_for(_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, do_search, req)
            if not fut.done():
                fut.set_result(result)
        except Exception as e:
            if not fut.done():
                fut.set_result({"ok": False, "error": "internal", "detail": str(e)})
        finally:
            touch_activity()
            _queue.task_done()


async def idle_watchdog() -> None:
    global _shutting_down
    while not _shutting_down:
        await asyncio.sleep(5)
        idle = time.time() - _last_active
        if idle >= IDLE_SEC and (_queue is None or _queue.empty()):
            log(f"Idle {int(idle)}s >= {IDLE_SEC}s → shutdown")
            _shutting_down = True
            break


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        data = await asyncio.wait_for(reader.readline(), timeout=30)
        if not data:
            writer.close()
            await writer.wait_closed()
            return
        raw = data.decode("utf-8", errors="replace").strip()
        resp = await handle_request(raw)
        writer.write((resp + "\n").encode("utf-8"))
        await writer.drain()
    except Exception as e:
        try:
            writer.write((json.dumps({"ok": False, "error": "handler", "detail": str(e)}) + "\n").encode())
            await writer.drain()
        except Exception:
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def write_pid() -> None:
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def cleanup_files() -> None:
    for p in (PID_FILE, SOCKET_PATH):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


async def run_server() -> None:
    global _queue, _shutting_down
    _queue = asyncio.Queue(maxsize=QUEUE_MAX)
    touch_activity()
    write_pid()
    load_model()

    if USE_TCP:
        server = await asyncio.start_server(handle_client, HOST, PORT)
        log(f"Listening TCP {HOST}:{PORT}")
    else:
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
            except Exception:
                pass
        server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
        try:
            os.chmod(SOCKET_PATH, 0o666)
        except Exception:
            pass
        log(f"Listening Unix socket {SOCKET_PATH}")

    workers = [asyncio.create_task(worker()) for _ in range(1)]
    watchdog = asyncio.create_task(idle_watchdog())

    log("Ready")
    try:
        while not _shutting_down:
            await asyncio.sleep(0.5)
    finally:
        _shutting_down = True
        watchdog.cancel()
        for w in workers:
            w.cancel()
        server.close()
        await server.wait_closed()
        cleanup_files()
        log("Stopped")


def acquire_lock() -> bool:
    try:
        import fcntl
        fd = open(LOCK_FILE, "w")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fd.close()
            return False
        fd.write(str(os.getpid()))
        fd.flush()
        globals()["_lock_fd"] = fd
        return True
    except ImportError:
        if os.path.exists(PID_FILE):
            try:
                pid = int(Path(PID_FILE).read_text().strip())
                os.kill(pid, 0)
                return False
            except Exception:
                pass
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()

    if args.status:
        try:
            from client_ipc import send_request
            print(json.dumps(send_request({"action": "stats"}), indent=2))
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}))
        return

    if args.stop:
        try:
            from client_ipc import send_request
            print(json.dumps(send_request({"action": "shutdown"}), indent=2))
        except Exception as e:
            if os.path.exists(PID_FILE):
                try:
                    pid = int(Path(PID_FILE).read_text().strip())
                    os.kill(pid, signal.SIGTERM)
                    print(json.dumps({"ok": True, "killed_pid": pid}))
                except Exception as e2:
                    print(json.dumps({"ok": False, "error": str(e2)}))
            else:
                print(json.dumps({"ok": False, "error": str(e)}))
        return

    if not acquire_lock():
        log("Another daemon is already running")
        sys.exit(0)

    def _sig(_s, _f):
        global _shutting_down
        _shutting_down = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
