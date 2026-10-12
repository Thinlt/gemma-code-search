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
import signal
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SOCKET_PATH = os.getenv("EMBEDDINGGEMMA_SOCK", "/tmp/embeddinggemma.sock")
PID_FILE = os.getenv("EMBEDDINGGEMMA_PID", "/tmp/embeddinggemma.daemon.pid")
LOCK_FILE = os.getenv("EMBEDDINGGEMMA_LOCK", "/tmp/embeddinggemma.daemon.lock")
IDLE_SEC = int(os.getenv("EMBEDDINGGEMMA_IDLE_SEC", "900"))
QUEUE_MAX = int(os.getenv("EMBEDDINGGEMMA_QUEUE_MAX", "24"))
INDEX_CACHE_MAX = int(os.getenv("EMBEDDINGGEMMA_INDEX_CACHE", "3"))
MAX_PER_FILE = int(os.getenv("EMBEDDINGGEMMA_MAX_PER_FILE", "2"))
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
    import torch
    from sentence_transformers import SentenceTransformer

    log("Loading EmbeddingGemma-2 (text-only)...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    _model = SentenceTransformer(
        "google/embeddinggemma-2",
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
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

    # diversity: without a per-file cap, one file whose many chunks all match
    # (e.g. an API client for exactly this query) floods the whole top-k and
    # pushes every other relevant file out of view
    max_per_file = int(req.get("max_per_file", MAX_PER_FILE))
    scan_limit = max(top_k * 20, 100)
    per_file: Dict[str, int] = {}
    picked: List[int] = []
    for idx in top_idx[:scan_limit]:
        score = float(scores[int(idx)])
        if score < min_score:
            break  # scores are descending: no later candidate can pass
        path = documents[int(idx)].get("path") or ""
        if max_per_file > 0 and per_file.get(path, 0) >= max_per_file:
            continue
        per_file[path] = per_file.get(path, 0) + 1
        picked.append(int(idx))
        if len(picked) >= top_k:
            break

    results = []
    for idx in picked:
        score = float(scores[idx])
        doc = documents[int(idx)]
        results.append({
            "path": doc.get("path"),
            "name": doc.get("name"),
            "kind": doc.get("kind"),
            "parent": doc.get("parent"),
            "start_line": doc.get("start_line"),
            "end_line": doc.get("end_line"),
            "language": doc.get("language"),
            "score": round(score, 4),
            "content": doc.get("content", "")[:2500],
        })

    return {
        "ok": True,
        "results": results,
        "indexed_commit": meta.get("last_commit", ""),
        "dirty_at_index": meta.get("dirty_at_index", []),
        "num_chunks": meta.get("num_chunks", len(documents)),
    }


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
            f"_To refresh those files: `python scripts/index.py --paths <file>`_\n"
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
