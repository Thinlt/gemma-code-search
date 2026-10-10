#!/usr/bin/env python3
"""
IPC client for EmbeddingGemma daemon.
Auto-starts daemon if not running.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SOCKET_PATH = os.getenv("EMBEDDINGGEMMA_SOCK", "/tmp/embeddinggemma.sock")
PID_FILE = os.getenv("EMBEDDINGGEMMA_PID", "/tmp/embeddinggemma.daemon.pid")
HOST = os.getenv("EMBEDDINGGEMMA_HOST", "127.0.0.1")
PORT = int(os.getenv("EMBEDDINGGEMMA_PORT", "8765"))
USE_TCP = os.getenv("EMBEDDINGGEMMA_TCP", "0") == "1" or sys.platform == "win32"

CONNECT_TIMEOUT = 0.4
START_WAIT_SEC = 120
REQUEST_TIMEOUT = 90


def _daemon_script() -> Path:
    return Path(__file__).resolve().parent / "daemon.py"


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def daemon_seems_running() -> bool:
    if os.path.exists(PID_FILE):
        try:
            pid = int(Path(PID_FILE).read_text().strip())
            if _is_pid_alive(pid):
                return True
        except Exception:
            pass
    if not USE_TCP and os.path.exists(SOCKET_PATH):
        return True
    return False


def _connect() -> socket.socket:
    if USE_TCP:
        s = socket.create_connection((HOST, PORT), timeout=CONNECT_TIMEOUT)
    else:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(CONNECT_TIMEOUT)
        s.connect(SOCKET_PATH)
    s.settimeout(REQUEST_TIMEOUT)
    return s


def _try_request(req: Dict[str, Any]) -> Dict[str, Any]:
    s = _connect()
    try:
        payload = (json.dumps(req) + "\n").encode("utf-8")
        s.sendall(payload)
        chunks = []
        while True:
            data = s.recv(65536)
            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                break
        raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
        return json.loads(raw)
    finally:
        try:
            s.close()
        except Exception:
            pass


def start_daemon() -> None:
    script = _daemon_script()
    if not script.exists():
        raise RuntimeError(f"daemon.py not found at {script}")

    env = os.environ.copy()
    kwargs: Dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore

    subprocess.Popen([sys.executable, str(script)], **kwargs)


def ensure_daemon(verbose: bool = True) -> None:
    try:
        _try_request({"action": "ping"})
        return
    except Exception:
        pass

    if verbose:
        print("[client] Daemon not running → starting (first load may take a while)...", flush=True)

    start_daemon()
    deadline = time.time() + START_WAIT_SEC
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        time.sleep(0.5)
        try:
            resp = _try_request({"action": "ping"})
            if resp.get("ok"):
                if verbose:
                    print("[client] Daemon ready.", flush=True)
                return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Daemon failed to become ready within {START_WAIT_SEC}s: {last_err}")


def send_request(req: Dict[str, Any], auto_start: bool = True, verbose: bool = True) -> Dict[str, Any]:
    if auto_start:
        ensure_daemon(verbose=verbose)
    else:
        return _try_request(req)

    try:
        return _try_request(req)
    except Exception:
        if verbose:
            print("[client] Connection lost → restarting daemon...", flush=True)
        start_daemon()
        time.sleep(1)
        ensure_daemon(verbose=verbose)
        return _try_request(req)
