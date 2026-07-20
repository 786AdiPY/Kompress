"""Benchmark every model in artifacts/models.json: latency + accuracy for the
native model and each compressed variant, using each model's own test data.
Writes per-model artifacts/<name>/benchmark_results.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config, feature_names            # noqa: E402
from common.metrics import compute_metrics               # noqa: E402
from common import onnx_utils                             # noqa: E402
from adapters import load_model                           # noqa: E402

WARMUP = int(os.getenv("WARMUP", "20"))
RUNS   = int(os.getenv("RUNS", "200"))


def load_index(cfg):
    with open(cfg.models_index) as f:
        return json.load(f)


def load_test(manifest):
    df = pd.read_csv(manifest["test_csv"])
    names = feature_names(manifest["features"])
    X = df[names].values.astype(np.float32)
    y = df[manifest["target"]].values
    return X, y


def _timed(fn, X):
    for _ in range(WARMUP):
        fn(X[:1])
    t0 = time.perf_counter()
    for _ in range(RUNS):
        out = fn(X)
    return out, (time.perf_counter() - t0) / RUNS * 1000


def _result(name, variant, latency, metrics):
    r = {"model": name, "kind": variant["kind"],
         "latency_ms": round(latency, 3), "model_size_kb": variant.get("size_kb", 0.0)}
    if variant.get("note"):
        r["note"] = variant["note"]
    r.update(metrics)
    return r


def bench_native(manifest, X, y):
    adapter = load_model(manifest["framework"], manifest["native"]["path"],
                         task=manifest["task"], num_classes=manifest["num_classes"])
    proba, latency = _timed(adapter.predict, X)
    return _result(manifest["native"]["name"], manifest["native"], latency,
                   compute_metrics(manifest["task"], y, proba))


def bench_onnx(variant, task, X, y):
    sess = onnx_utils.make_session(variant["path"])
    fn = lambda batch: onnx_utils.extract_proba(onnx_utils.run(sess, batch), task)
    proba, latency = _timed(fn, X)
    return _result(variant["name"], variant, latency, compute_metrics(task, y, proba))


def bench_trt(variant, task, X, y):
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
    except Exception as e:
        print(f"[{variant['name']}] TRT runtime unavailable ({e}); skipping.")
        return None
    with open(variant["path"], "rb") as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    def infer(batch):
        out = np.empty((len(batch),), dtype=np.float32)
        d_in = cuda.mem_alloc(batch.nbytes)
        d_out = cuda.mem_alloc(out.nbytes)
        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_in, np.ascontiguousarray(batch), stream)
        context.execute_async_v2([int(d_in), int(d_out)], stream.handle)
        cuda.memcpy_dtoh_async(out, d_out, stream)
        stream.synchronize()
        return out

    proba, latency = _timed(infer, X)
    return _result(variant["name"], variant, latency, compute_metrics(task, y, proba))


def benchmark_model(manifest) -> list:
    X, y = load_test(manifest)
    task = manifest["task"]
    print(f"\n=== Benchmarking '{manifest['name']}'  {len(X)} rows, "
          f"{X.shape[1]} features, task={task} ===")

    results = [bench_native(manifest, X, y)]
    for v in manifest["variants"]:
        if v["kind"] == "onnx":
            r = bench_onnx(v, task, X, y)
        elif v["kind"] == "trt":
            r = bench_trt(v, task, X, y)
        else:
            r = bench_native(manifest, X, y)
        if r is not None:
            results.append(r)

    baseline_lat = results[0]["latency_ms"]
    for r in results:
        r["speedup_vs_native"] = round(baseline_lat / r["latency_ms"], 2) if r["latency_ms"] else 1.0

    print("{:<26} {:>12} {:>12} {:>10}".format("variant", "latency_ms", "size_kb", "speedup"))
    print("-" * 64)
    for r in results:
        print("{:<26} {:>12.3f} {:>12.1f} {:>9.2f}x".format(
            r["model"], r["latency_ms"], r["model_size_kb"], r["speedup_vs_native"]))
    return results


def main():
    cfg = load_config()
    index = load_index(cfg)
    for entry in index["models"]:
        with open(entry["manifest"]) as f:
            manifest = json.load(f)
        results = benchmark_model(manifest)
        out = os.path.join(os.path.dirname(entry["manifest"]), "benchmark_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results -> {out}")


if __name__ == "__main__":
    main()
