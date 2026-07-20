"""Framework-agnostic, MULTI-model FastAPI serving.

Reads artifacts/models.json and serves every model + every compressed variant:
  POST /predict/{model}/{variant}
  POST /compare/{model}
  GET  /models
  GET  /health
Each model's request schema and task behavior come from its own manifest.
"""
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import build_request_model, records_to_array  # noqa: E402
from common import onnx_utils                              # noqa: E402

MODELS_INDEX = os.getenv("MODELS_INDEX", "artifacts/models.json")

MODELS: dict[str, dict] = {}   # model_name -> {manifest, request_model, task, features, loaded}


def _load_variant(variant, manifest):
    kind = variant["kind"]
    if kind == "onnx":
        return ("onnx", onnx_utils.make_session(variant["path"]))
    if kind == "native":
        from adapters import load_model
        return ("native", load_model(manifest["framework"], variant["path"],
                                     task=manifest["task"], num_classes=manifest["num_classes"]))
    if kind == "trt":
        try:
            import tensorrt as trt
            with open(variant["path"], "rb") as f:
                engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
            return ("trt", engine)
        except Exception:
            fp32 = next((v for v in manifest["variants"] if v["name"] == "onnx_fp32"), None)
            if fp32:
                return ("onnx", onnx_utils.make_session(fp32["path"]))
            raise
    raise ValueError(f"Unknown variant kind: {kind}")


def _load_model(manifest):
    loaded = {}
    for v in [manifest["native"], *manifest["variants"]]:
        try:
            loaded[v["name"]] = _load_variant(v, manifest)
        except Exception as e:
            print(f"[warn] {manifest['name']}/{v['name']} failed to load: {e}")
    loaded.setdefault("native", loaded.get(manifest["native"]["name"]))
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODELS_INDEX) as f:
        index = json.load(f)
    for entry in index["models"]:
        with open(entry["manifest"]) as f:
            manifest = json.load(f)
        loaded = _load_model(manifest)
        if not loaded:
            print(f"[warn] no servable variants for {manifest['name']}; skipping.")
            continue
        MODELS[manifest["name"]] = {
            "manifest": manifest, "task": manifest["task"], "features": manifest["features"],
            "request_model": build_request_model(manifest["features"], f"{manifest['name']}_Request"),
            "loaded": loaded,
        }
        print(f"Loaded '{manifest['name']}' [{manifest['framework']}/{manifest['task']}] "
              f"variants: {list(loaded.keys())}")
    if not MODELS:
        raise RuntimeError("No models could be loaded from the index.")
    yield
    MODELS.clear()


app = FastAPI(title="Multi-Model Compression API", version="3.0.0", lifespan=lifespan)


class PredictResponse(BaseModel):
    model: str
    variant: str
    prediction: Any
    probability: Any = None
    latency_ms: float


class CompareResponse(BaseModel):
    model: str
    input: dict
    results: List[PredictResponse]


def _get_model(name):
    if name not in MODELS:
        raise HTTPException(404, f"Model '{name}' not found. Available: {list(MODELS)}")
    return MODELS[name]


def _parse(entry, payload: dict):
    try:
        return entry["request_model"](**payload)
    except ValidationError as e:
        raise HTTPException(422, [{"loc": list(x["loc"]), "msg": x["msg"], "type": x["type"]}
                                  for x in e.errors()])


def _infer(kind_model, X, task):
    kind, model = kind_model
    if kind == "native":
        return np.asarray(model.predict(X))
    if kind == "onnx":
        return np.asarray(onnx_utils.extract_proba(onnx_utils.run(model, X), task))
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa
    context = model.create_execution_context()
    out = np.empty((len(X),), dtype=np.float32)
    d_in, d_out = cuda.mem_alloc(X.nbytes), cuda.mem_alloc(out.nbytes)
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_in, np.ascontiguousarray(X), stream)
    context.execute_async_v2([int(d_in), int(d_out)], stream.handle)
    cuda.memcpy_dtoh_async(out, d_out, stream)
    stream.synchronize()
    return out


def _format(proba, task):
    if task == "regression":
        return float(proba.ravel()[0]), None
    if task == "binary_classification":
        p = float(proba.ravel()[0])
        return int(p > 0.5), round(p, 6)
    row = proba[0] if proba.ndim == 2 else proba
    return int(np.argmax(row)), [round(float(x), 6) for x in row]


def _predict(model_name, variant, req) -> PredictResponse:
    entry = _get_model(model_name)
    if variant not in entry["loaded"]:
        raise HTTPException(404, f"Variant '{variant}' not in {model_name}. "
                                 f"Have: {list(entry['loaded'])}")
    X = records_to_array(req.model_dump(), entry["features"])
    t0 = time.perf_counter()
    proba = _infer(entry["loaded"][variant], X, entry["task"])
    latency = (time.perf_counter() - t0) * 1000
    pred, prob = _format(proba, entry["task"])
    return PredictResponse(model=model_name, variant=variant, prediction=pred,
                           probability=prob, latency_ms=round(latency, 3))


@app.post("/predict/{model}/{variant}", response_model=PredictResponse)
def predict(model: str, variant: str, payload: dict):
    entry = _get_model(model)
    return _predict(model, variant, _parse(entry, payload))


@app.post("/compare/{model}", response_model=CompareResponse)
def compare(model: str, payload: dict):
    entry = _get_model(model)
    req = _parse(entry, payload)
    manifest = entry["manifest"]
    names = [manifest["native"]["name"]] + [v["name"] for v in manifest["variants"]]
    results = [_predict(model, n, req) for n in names if n in entry["loaded"]]
    return CompareResponse(model=model, input=req.model_dump(), results=results)


@app.get("/models")
def models():
    return {m: {"task": e["task"], "variants": list(e["loaded"].keys()),
                "features": [f["name"] for f in e["features"]]}
            for m, e in MODELS.items()}


@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODELS.keys())}
