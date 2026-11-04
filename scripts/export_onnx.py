#!/usr/bin/env python3
"""
Export google/embeddinggemma-2 to an int8-quantized ONNX bundle for the
optional ONNX backend (EMBEDDINGGEMMA_BACKEND=onnx / auto).

Pipeline: sentence-transformers fp32 reference -> torch.onnx.export (dynamo,
legacy fallback) -> dynamic int8 quantization -> validation vs the ST
pipeline -> bundle.json (written only when validation passes).

Usage:
  python scripts/export_onnx.py [--samples 48] [--keep-fp32]

Validation compares, on real chunks when a local index exists:
  - per-chunk cosine between ST fp32 and ONNX int8 embeddings
  - top-5 retrieval overlap for a handful of queries
Thresholds gate bundle.json: mean cosine >= 0.97, min >= 0.90, overlap >= 0.6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
BUNDLE_DIR = SCRIPTS.parent / "models" / "embeddinggemma2-int8"
FP32_ONNX = "model.onnx"
INT8_ONNX = "model.int8.onnx"

MIN_MEAN_COS = 0.97
MIN_MIN_COS = 0.90
MIN_OVERLAP = 0.60


def load_st():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import logging as hf_logging

    hf_logging.disable_progress_bar()  # drop the "Loading weights" bar — noise in logs

    m = SentenceTransformer(
        "google/embeddinggemma-2",
        model_kwargs={"torch_dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    m.max_seq_length = int(__import__("os").getenv("EMBEDDINGGEMMA_MAX_SEQ_LEN", "2048"))
    return m


def read_model_configs() -> dict:
    """Prompts + pooling straight from the HF cache (no model load)."""
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download("google/embeddinggemma-2", "config_sentence_transformers.json")) as f:
        st_cfg = json.load(f)
    with open(hf_hub_download("google/embeddinggemma-2", "1_Pooling/config.json")) as f:
        pooling = json.load(f)
    return {"prompts": st_cfg.get("prompts", {}), "pooling": pooling}


def export_onnx(st, out_dir: Path) -> None:
    import torch

    net = getattr(st[0], "auto_model", None) or st[0].model
    net = net.eval().cpu()

    class _TextOnly(torch.nn.Module):
        # the source model is multimodal-capable: forward(input_ids, pixel_values, ...)
        # would swallow a positional attention_mask into pixel_values — force keywords
        def __init__(self, m):
            super().__init__()
            self.net = m

        def forward(self, input_ids, attention_mask):
            return self.net(input_ids=input_ids, attention_mask=attention_mask)[0]

    wrapper = _TextOnly(net).eval()
    ids = torch.ones(1, 16, dtype=torch.int64)
    mask = torch.ones(1, 16, dtype=torch.int64)
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    }
    try:
        torch.onnx.export(
            wrapper, (ids, mask), str(out_dir / FP32_ONNX),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            dynamo=True,
        )
        print("exported via dynamo exporter")
    except Exception as e:
        print(f"dynamo export failed ({e}); retrying legacy exporter...")
        torch.onnx.export(
            wrapper, (ids, mask), str(out_dir / FP32_ONNX),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        print("exported via legacy exporter")


def quantize(out_dir: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # MatMul weights only: quantizing the 262k-token embedding table (Gather) is
    # what drags cosine similarity down on this model
    quantize_dynamic(
        str(out_dir / FP32_ONNX),
        str(out_dir / INT8_ONNX),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    print(f"quantized -> {INT8_ONNX}")


def sample_texts(n: int) -> list:
    """Real chunks from the local index when present, else synthetic code."""
    import os
    import pickle

    idx = Path(os.getenv("CODE_INDEX_DIR", "./.code_index")) / "documents.pkl"
    if idx.exists():
        with open(idx, "rb") as f:
            docs = pickle.load(f)
        step = max(1, len(docs) // n)
        texts = [d.get("content", "") for d in docs[::step]][:n]
        if texts:
            print(f"validating on {len(texts)} real chunks from {idx}")
            return texts
    print("no local index found; validating on synthetic snippets")
    return [
        "def refresh_token(session, token):\n    if expired(token):\n        return rotate(session, token)\n    return token\n" * 3
    ] * min(n, 8)


def validate(st, onnx_emb, samples: int) -> dict:
    texts = sample_texts(samples)
    t0 = time.time()
    a = st.encode(
        texts, prompt_name="Document", truncate_dim=512, normalize_embeddings=True,
        show_progress_bar=False, batch_size=4,
    )
    t_st = time.time() - t0
    time.sleep(3)
    t0 = time.time()
    b = onnx_emb.encode(
        texts, prompt_name="Document", truncate_dim=512, normalize_embeddings=True, batch_size=4
    )
    t_onnx = time.time() - t0
    a, b = np.asarray(a), np.asarray(b)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

    queries = [
        "rate limiting middleware",
        "refresh token rotation",
        "xml document validation",
        "pdf report rendering",
        "webhook delivery retry",
        "database migration rollback",
    ]
    qa = np.asarray(st.encode(
        queries, prompt_name="CodeRetrieval", truncate_dim=512, normalize_embeddings=True,
        show_progress_bar=False, batch_size=4,
    ))
    time.sleep(3)
    qb = np.asarray(onnx_emb.encode(
        queries, prompt_name="CodeRetrieval", truncate_dim=512, normalize_embeddings=True, batch_size=4
    ))
    top_a = np.argsort(a @ qa.T, axis=0)[::-1][:5]
    top_b = np.argsort(b @ qb.T, axis=0)[::-1][:5]
    overlap = float(np.mean([len(set(top_a[:, i]) & set(top_b[:, i])) / 5.0 for i in range(len(queries))]))

    return {
        "samples": len(texts),
        "cos_mean": float(cos.mean()),
        "cos_min": float(cos.min()),
        "top5_overlap": overlap,
        "sec_st_fp32": round(t_st, 2),
        "sec_onnx_int8": round(t_onnx, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EmbeddingGemma-2 int8 ONNX bundle")
    parser.add_argument("--samples", type=int, default=48, help="chunks used for validation")
    parser.add_argument("--keep-fp32", action="store_true", help="Keep the fp32 model.onnx next to the int8 one")
    args = parser.parse_args()

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sentence-transformers reference (fp32)...")
    st = load_st()
    cfg = read_model_configs()
    if cfg["pooling"].get("pooling_mode") != "mean":
        sys.exit(f"[error] pooling_mode={cfg['pooling'].get('pooling_mode')} not supported by onnx_embedder")

    export_onnx(st, BUNDLE_DIR)
    quantize(BUNDLE_DIR)
    if not args.keep_fp32:
        # dynamo externalizes fp32 weights to model.onnx.data — remove both
        # (the int8 graph is self-contained; quantize_dynamic inlines everything)
        (BUNDLE_DIR / FP32_ONNX).unlink(missing_ok=True)
        (BUNDLE_DIR / f"{FP32_ONNX}.data").unlink(missing_ok=True)

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained("google/embeddinggemma-2").save_pretrained(BUNDLE_DIR)
    print("tokenizer saved")

    # provisional bundle so OnnxEmbedder can load; rewritten with validation stats
    # below, removed again if validation fails
    bundle = {
        "version": 1,
        "dim": 768,
        "pooling": "mean",
        "include_prompt": True,
        "int8": True,
        "max_seq_length": st.max_seq_length,
        "prompts": cfg["prompts"],
        "validation": None,
    }
    with open(BUNDLE_DIR / "bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)

    from onnx_embedder import OnnxEmbedder

    emb = OnnxEmbedder(bundle_dir=BUNDLE_DIR)
    print(f"onnx session ready (provider={emb.provider})")
    stats = validate(st, emb, args.samples)
    print(json.dumps(stats, indent=2))

    if stats["cos_mean"] < MIN_MEAN_COS or stats["cos_min"] < MIN_MIN_COS or stats["top5_overlap"] < MIN_OVERLAP:
        (BUNDLE_DIR / INT8_ONNX).unlink(missing_ok=True)
        (BUNDLE_DIR / "bundle.json").unlink(missing_ok=True)
        sys.exit(
            "[error] int8 validation failed thresholds "
            f"(mean>={MIN_MEAN_COS}, min>={MIN_MIN_COS}, overlap>={MIN_OVERLAP}) — bundle removed"
        )

    bundle["validation"] = stats
    with open(BUNDLE_DIR / "bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"bundle written: {BUNDLE_DIR / 'bundle.json'} — activate with EMBEDDINGGEMMA_BACKEND=onnx (torch stays default: int8 measured slower on GPU machines)")


if __name__ == "__main__":
    main()
