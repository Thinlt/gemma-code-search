#!/usr/bin/env python3
"""Resident Core AI worker for the ANEMLL text tower (embeddinggemma-2 ANE).

Runs inside the python3.13 venv that has ``coreai-core`` installed; the host
process (skill venv, python 3.12) talks over stdin/stdout JSON lines and
passes feeds/results through .npz/.npy files, mirroring the upstream
Anemll/anemll-embeddings worker protocol (text tower only).

Commands:
  load    {"package": <.aimodel dir>, "compute": "ane"|"cpu"} -> placement report
  embed   {"npz": <path with embeds [N,320,512] f16 + mask [N,320] f16>,
           "out": <path for [N,768] f16>}                     -> per-batch latency
  shutdown
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SUPPORTED_COMPUTE = ("ane", "cpu")


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


def _coreai_python_candidates() -> list[str]:
    # sibling venv inside the model bundle, then the upstream default layout
    bundle = Path(__file__).resolve().parent.parent / "models" / "embeddinggemma2-ane"
    env_override = os.environ.get("EMBEDDINGGEMMA_ANE_PYTHON") or os.environ.get("ANEMLL_COREAI_PYTHON")
    return [
        env_override,
        str(bundle / "coreai-venv" / "bin" / "python"),
        str(Path.home() / ".anemll-embeddings" / "coreai-venv" / "bin" / "python"),
        "/opt/homebrew/bin/python3.13",
        "/usr/local/bin/python3.13",
    ]


def _reexec_if_needed() -> None:
    try:
        import coreai.runtime  # noqa: F401

        return
    except ImportError:
        pass
    for exe in _coreai_python_candidates():
        if exe and Path(exe).is_file():
            os.execv(exe, [exe, *sys.argv])
    _reply({
        "ok": False,
        "error": "no python can import coreai.runtime; create models/embeddinggemma2-ane/coreai-venv "
        "with python3.13 and `pip install coreai-core==1.0.0b2 numpy`, or set EMBEDDINGGEMMA_ANE_PYTHON",
    })
    raise SystemExit(1)


def _specialization(compute: str):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    if not SpecializationOptions.is_supported():
        return None
    if compute == "ane":
        return SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.neural_engine()
        )
    return SpecializationOptions.cpu_only()


def _placement(model) -> str:
    for attr in ("placement", "compute_placement"):
        val = getattr(model, attr, None)
        if val is not None and "fullyOnANE" in str(val):
            return "fullyOnANE"
    devices: list[str] = []
    try:
        raw = model._debug_infos
        blob = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        devices = _devices_from(json.loads(blob))
    except Exception as exc:  # noqa: BLE001 - debug info is optional
        _log(f"debug_infos: {type(exc).__name__}: {exc}")
    if devices and all(d == "ANE" for d in devices):
        return "fullyOnANE"
    if devices:
        return "+".join(sorted(set(devices)))
    return "unknown"


def _devices_from(obj, *, limit: int = 4000) -> list[str]:
    found: list[str] = []
    keys = ("device", "computeunit", "compute_unit", "preferreddevice",
            "preferred_compute_device", "computeunitkind")

    def walk(node):
        if len(found) >= limit:
            return
        if isinstance(node, dict):
            for key, val in node.items():
                lk = str(key).strip().lower().replace("_", "")
                if lk in keys and isinstance(val, (str, int)):
                    s = str(val).strip().upper()
                    if "NEURAL" in s or "NEURALENGINE" in s.replace(" ", "") or s in ("ANE", "NE"):
                        found.append("ANE")
                    elif "GPU" in s or "METAL" in s:
                        found.append("GPU")
                    elif "CPU" in s:
                        found.append("CPU")
                walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found


class State:
    fn = None
    placement = "unknown"


async def _load(package: str, compute: str) -> dict:
    from coreai.runtime import AIModel, NDArray

    pkg = Path(package)
    if not pkg.is_dir():
        raise FileNotFoundError(f"missing package {pkg}")
    os.environ["ANEMLL_COREAI_COMPUTE"] = compute  # parity-runner convention
    t0 = time.perf_counter()
    opts = _specialization(compute)
    if opts is not None:
        model = await AIModel.load(pkg, specialization_options=opts)
    else:
        model = await AIModel.load(pkg)
    names = list(model.function_names)
    fn_name = "text_embeds_s320" if "text_embeds_s320" in names else names[0]
    State.fn = model.load_function(fn_name)
    load_ms = (time.perf_counter() - t0) * 1000.0
    State.placement = _placement(model)

    # warmup with a legal dummy feed so the first real embed is not compile time
    ins = {"inputs_embeds": NDArray(np.zeros((1, 320, 512), dtype=np.float16)),
           "attention_mask": NDArray(np.concatenate([np.ones((1, 1), np.float16),
                                                     np.zeros((1, 319), np.float16)], axis=1))}
    t1 = time.perf_counter()
    await State.fn(inputs=ins)
    return {
        "entry": fn_name,
        "load_ms": round(load_ms, 1),
        "warmup_ms": round((time.perf_counter() - t1) * 1000.0, 1),
        "placement": State.placement,
        "specialization": compute,
    }


async def _embed(npz_path: str, out_path: str) -> dict:
    from coreai.runtime import NDArray

    if State.fn is None:
        raise RuntimeError("tower not loaded")
    with np.load(npz_path) as data:
        embeds = data["embeds"]
        mask = data["mask"]
    t0 = time.perf_counter()
    outs = []
    for i in range(embeds.shape[0]):
        feed = {
            "inputs_embeds": NDArray(np.ascontiguousarray(embeds[i : i + 1], dtype=np.float16)),
            "attention_mask": NDArray(np.ascontiguousarray(mask[i : i + 1], dtype=np.float16)),
        }
        result = await State.fn(inputs=feed)
        first = next(iter(result))
        outs.append(result[first].numpy()[0])  # drop the batch dim -> [768]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    arr = np.stack(outs, axis=0)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr.astype(np.float16, copy=False))
    return {"n": int(arr.shape[0]), "latency_ms": round(elapsed_ms, 1)}


async def _serve() -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "load":
                compute = (req.get("compute") or "ane").strip().lower()
                if compute not in SUPPORTED_COMPUTE:
                    raise ValueError(f"compute must be one of {SUPPORTED_COMPUTE}")
                _reply({"ok": True, **await _load(req["package"], compute)})
            elif cmd == "embed":
                _reply({"ok": True, **await _embed(req["npz"], req["out"])})
            elif cmd == "shutdown":
                _reply({"ok": True})
                break
            else:
                _reply({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:  # noqa: BLE001 - the worker must reply, not die
            _log(f"worker error: {type(exc).__name__}: {exc}")
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    _reexec_if_needed()
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
