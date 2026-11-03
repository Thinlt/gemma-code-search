#!/usr/bin/env python3
"""ANEMLL Core AI backend for EmbeddingGemma-2 (anemll/anemll-embeddinggemma-2-ane).

Same encode() contract as OnnxEmbedder: prompt prepend -> tokenize -> tower ->
Matryoshka slice -> L2 normalize. The transformer+pooling+projection run as a
fixed-shape fp16 Core AI package on the Neural Engine
(models/embeddinggemma2-ane/text_embeds_s320.aimodel); this host side does the
Gemma token lookup (embed_tokens x sqrt(512)), truncates/pads to the package's
320-token window, and drives a resident python3.13 worker
(scripts/anemll_worker.py) that owns the Core AI runtime.

Text longer than 320 tokens is hard-truncated by the package contract — the
chunker's TOKEN_LIMIT=320 exists precisely so indexed chunks never hit that
cut.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import List, Union

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parent.parent / "models" / "embeddinggemma2-ane"
TOWER_DIR = BUNDLE_DIR / "text_embeds_s320.aimodel"
HOST_DIR = BUNDLE_DIR / "host"
SEQ_LEN = 320           # package contract: inputs_embeds [1, 320, 512] fp16
HIDDEN = 512
EMBED_SCALE = HIDDEN ** 0.5  # Gemma scales token rows by sqrt(hidden)
EMBED_WEIGHT_KEY = "embed_tokens.weight"


def _coreai_python() -> Path:
    override = os.getenv("EMBEDDINGGEMMA_ANE_PYTHON") or os.getenv("ANEMLL_COREAI_PYTHON")
    candidates = [
        override,
        BUNDLE_DIR / "coreai-venv" / "bin" / "python",
        Path.home() / ".anemll-embeddings" / "coreai-venv" / "bin" / "python",
    ]
    for exe in candidates:
        if exe and Path(exe).is_file():
            return Path(exe)
    raise RuntimeError(
        "no Core AI python found; create models/embeddinggemma2-ane/coreai-venv with "
        "python3.13 and `pip install coreai-core==1.0.0b2 numpy`, or set EMBEDDINGGEMMA_ANE_PYTHON"
    )


def bundle_available() -> bool:
    """True when the tower + host files exist and a Core AI interpreter is set up."""
    if not all(p.is_file() for p in (
        TOWER_DIR / "main.mlirb",
        HOST_DIR / "embed_tokens.safetensors",
        HOST_DIR / "tokenizer.json",
    )):
        return False
    try:
        _coreai_python()
    except RuntimeError:
        return False
    return True


def _parse_safetensors(path: Path) -> dict:
    raw = path.read_bytes()
    header_len = struct.unpack_from("<Q", raw, 0)[0]
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    return {"header": header, "data": raw, "header_len": header_len}


def _bf16_to_fp16(buf: bytes) -> np.ndarray:
    """bf16 bytes -> fp16 numpy (lossless for embedding-row magnitudes)."""
    u16 = np.frombuffer(buf, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).astype(np.float16)


def _explain(error: str) -> str:
    """Translate opaque Core AI failures into something actionable.

    When a Neural Engine program can't be mapped, Core AI surfaces the real
    cause as `Program load failed — no memory (transient; retry under lower
    memory pressure)`, but only when no compute unit is preferred. With
    `ComputeUnitKind.neural_engine()` it collapses to
    `Foundation._GenericObjCError error 0.` — which tells the user nothing.
    """
    if "GenericObjCError" in error or "no memory" in error:
        return (
            f"{error} — Neural Engine could not load the tower; the usual cause is "
            "memory pressure (ANE program mapping needs free RAM). Close memory-hungry "
            "apps and retry, or force another backend: "
            "EMBEDDINGGEMMA_ANE_COMPUTE=cpu / EMBEDDINGGEMMA_BACKEND=torch"
        )
    return error


class AnemllEmbedder:
    """SentenceTransformer-compatible encode() over the ANE Core AI tower."""

    encoder_tag = "anemll-ane"

    def __init__(self, bundle_dir=None):
        from transformers import AutoTokenizer

        host = Path(bundle_dir) / "host" if bundle_dir else HOST_DIR
        with open(host / "config_sentence_transformers.json") as f:
            self.prompts = json.load(f).get("prompts", {})
        self.tokenizer = AutoTokenizer.from_pretrained(str(host))
        self.max_seq_length = SEQ_LEN
        self.dim = 768
        self.device = SimpleNamespace(type="anemll")
        self.placement = "unknown"
        self._lock = threading.Lock()
        self._tmp = tempfile.TemporaryDirectory(prefix="anemll-skill-")
        self._seq = 0

        st = _parse_safetensors(host / "embed_tokens.safetensors")
        info = st["header"][EMBED_WEIGHT_KEY]
        start, end = info["data_offsets"]
        table = _bf16_to_fp16(st["data"][8 + st["header_len"] + start : 8 + st["header_len"] + end])
        self._table = table.reshape(info["shape"])
        # Gemma's <pad> row contributes nothing (mask 0 wherever it appears)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        if 0 <= pad_id < self._table.shape[0]:
            self._table[pad_id] = 0

        tower = Path(bundle_dir) / "text_embeds_s320.aimodel" if bundle_dir else TOWER_DIR
        compute = os.getenv("EMBEDDINGGEMMA_ANE_COMPUTE", "ane").strip().lower()
        self._load_payload = {"cmd": "load", "package": str(tower), "compute": compute}
        self._proc = self._spawn()
        try:
            report = self._request(self._load_payload)
        except BaseException:
            # don't strand a live worker/temp dir when the tower can't be loaded
            # (e.g. Neural Engine out of memory); the caller falls back to torch
            self.close()
            raise
        self.placement = report.get("placement", "unknown")
        self.provider = f"coreai:{compute}:{self.placement}"

    # ---- worker IPC ------------------------------------------------------
    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            [str(_coreai_python()), str(Path(__file__).resolve().parent / "anemll_worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # worker logs are chatty; failures come back as replies
            text=True,
            bufsize=1,
        )

    def _recover(self) -> None:
        """Replace a dead worker and reload the tower into it.

        Reloading is the point: a fresh worker has no tower, so retrying an
        `embed` without re-issuing `load` would only earn "tower not loaded".
        The temp dir is deliberately left alone — `close()` deletes it, and
        `_encode_batch` still needs it for the retried request.
        """
        old = self._proc
        self._proc = self._spawn()
        if old is not None and old.poll() is None:
            old.terminate()
        self._request(self._load_payload, _retry=False)

    def _request(self, payload: dict, _retry: bool = True) -> dict:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("coreai worker is not running")

        def recover_or_raise(msg: str):
            # only mid-session commands are worth recovering: re-issuing a failed
            # `load` would just hit the same wall (e.g. ANE out of memory), and
            # recovering a `shutdown` would spin up a worker purely to kill it
            if _retry and payload.get("cmd") not in ("load", "shutdown"):
                self._recover()
                return self._request(payload, _retry=False)
            raise RuntimeError(msg)

        # make sure the process is still alive
        if proc.poll() is not None:
            return recover_or_raise("coreai worker process died")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            # empty reply = worker died without answering
            return recover_or_raise("coreai worker died without replying")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(_explain(reply.get("error") or "coreai worker failed"))
        return reply

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._request({"cmd": "shutdown"})
                except Exception:
                    self._proc.terminate()
            self._proc = None
            self._tmp.cleanup()

    # ---- encode ----------------------------------------------------------
    def encode(
        self,
        texts: Union[str, List[str]],
        prompt_name: str = None,
        truncate_dim: int = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        batch_size: int = 16,
        **_,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        prompt = self.prompts.get(prompt_name or "", "")

        with self._lock:
            out: List[np.ndarray] = []
            bs = max(1, batch_size)
            for i in range(0, len(texts), bs):
                out.append(self._encode_batch(
                    [prompt + t for t in texts[i : i + bs]] if prompt else texts[i : i + bs]
                ))
        emb = np.concatenate(out, axis=0)[:, : (truncate_dim or self.dim)].astype(np.float32)
        if normalize_embeddings:
            emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return emb[0] if single else emb

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        # per-text tokenize without padding, then truncate to the package window
        ids_list, masks_list = [], []
        for text in texts:
            enc = self.tokenizer(text, return_tensors="np", padding=False, truncation=False)
            ids = enc["input_ids"][0][:SEQ_LEN]
            mask = enc["attention_mask"][0][:SEQ_LEN]
            ids_list.append(ids)
            masks_list.append(mask)
        n = len(texts)
        embeds = np.zeros((n, SEQ_LEN, HIDDEN), dtype=np.float16)
        mask = np.zeros((n, SEQ_LEN), dtype=np.float16)
        for row, (ids, m) in enumerate(zip(ids_list, masks_list)):
            # token lookup + Gemma sqrt(hidden) scale; scale in fp32 so the
            # fp16 cast happens at the same point as the upstream host path
            embeds[row, : len(ids)] = (self._table[ids].astype(np.float32) * EMBED_SCALE).astype(np.float16)
            mask[row, : len(m)] = m.astype(np.float16)

        self._seq += 1
        folder = Path(self._tmp.name)
        npz = folder / f"in_{self._seq}.npz"
        out = folder / f"out_{self._seq}.npy"
        try:
            np.savez(npz, embeds=embeds, mask=mask)
            self._request({"cmd": "embed", "npz": str(npz), "out": str(out)})
            return np.load(out).astype(np.float32)
        finally:
            npz.unlink(missing_ok=True)
            out.unlink(missing_ok=True)
