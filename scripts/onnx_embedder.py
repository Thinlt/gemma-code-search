#!/usr/bin/env python3
"""
Optional int8 ONNX backend for EmbeddingGemma-2 (produced by export_onnx.py).

Replicates the exact sentence-transformers pipeline of google/embeddinggemma-2:
prompt prepend -> tokenize -> transformer -> mean pooling with the prompt tokens
excluded (include_prompt=True) -> Matryoshka slice -> L2 normalize.

The bundle under models/embeddinggemma2-int8/ carries the prompts and pooling
config read from the source model at export time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Union

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parent.parent / "models" / "embeddinggemma2-int8"
BUNDLE_NAME = "bundle.json"
ONNX_NAME = "model.int8.onnx"
DEFAULT_MAX_SEQ_LEN = int(os.getenv("EMBEDDINGGEMMA_MAX_SEQ_LEN", "2048"))


def bundle_available() -> bool:
    """True when the exported int8 bundle exists and onnxruntime is importable."""
    if not (BUNDLE_DIR / BUNDLE_NAME).exists() or not (BUNDLE_DIR / ONNX_NAME).exists():
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def encoder_id(model, dim: int) -> str:
    """Identity of the encoder that produced/produces embeddings — index and
    queries must agree on it for cosine scores to mean anything."""
    if isinstance(model, OnnxEmbedder):
        return f"embeddinggemma-2:onnx-int8:{dim}"
    dt = str(next(model.parameters()).dtype).replace("torch.", "")
    return f"embeddinggemma-2:torch:{dt}:{dim}"


class OnnxEmbedder:
    """SentenceTransformer-compatible encode() over the quantized ONNX graph."""

    def __init__(self, bundle_dir=None, max_seq_length: int = DEFAULT_MAX_SEQ_LEN):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        bundle_dir = Path(bundle_dir) if bundle_dir is not None else BUNDLE_DIR

        with open(bundle_dir / BUNDLE_NAME) as f:
            self.bundle = json.load(f)
        self.prompts = self.bundle.get("prompts", {})
        if self.bundle.get("pooling") != "mean":
            raise RuntimeError(f"unsupported pooling mode: {self.bundle.get('pooling')}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir))
        self.tokenizer.padding_side = "right"
        self.max_seq_length = max_seq_length
        self.dim = int(self.bundle.get("dim", 768))

        ep = os.getenv("EMBEDDINGGEMMA_ONNX_EP", "auto").lower()
        available = ort.get_available_providers()
        if ep in ("", "auto", "cpu"):
            # int8 runs well on the CPU EP (neon dot); CoreML splits this graph
            # into hundreds of partitions — opt in explicitly with
            # EMBEDDINGGEMMA_ONNX_EP=coreml only after measuring it
            providers = ["CPUExecutionProvider"]
        else:
            providers = [ep] if ep in available else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(bundle_dir / ONNX_NAME), providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_names = {i.name for i in self.session.get_inputs()}
        outs = self.session.get_outputs()
        self.output_name = next((o.name for o in outs if o.name == "last_hidden_state"), outs[0].name)
        self.device = SimpleNamespace(type="onnx")

    def encode(
        self,
        texts: Union[str, List[str]],
        prompt_name: str = None,
        truncate_dim: int = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        batch_size: int = 8,
        **_,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        prompt = self.prompts.get(prompt_name or "", "")
        full = [prompt + t for t in texts] if prompt else list(texts)

        out: List[np.ndarray] = []
        bs = max(1, batch_size)
        for i in range(0, len(full), bs):
            enc = self.tokenizer(
                full[i : i + bs],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            feeds = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
            hidden = self.session.run([self.output_name], feeds)[0]
            # mean pooling over all attention-masked tokens — measured against
            # sentence-transformers 6.1: this model's processor path pools the
            # prompt tokens too (prompt_length is never set), despite the
            # bundle's include_prompt flag coming from the pooling config
            mask = enc["attention_mask"].astype(np.float32)
            summed = (hidden.astype(np.float32) * mask[..., None]).sum(axis=1)
            counted = np.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
            out.append(summed / counted)

        emb = np.concatenate(out, axis=0)[:, : (truncate_dim or self.dim)].astype(np.float32)
        if normalize_embeddings:
            emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return emb[0] if single else emb
