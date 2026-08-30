#!/usr/bin/env python3
"""Standalone empirical benchmark script for ONNX FP32 ResNet-50 model."""
import json
import os
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.abspath("api_runs/4886d76b30e64e9a93988eb19d07e132/artifacts/ResNet-50 ImageNet Benchmark/model_fp32.onnx")
DATA_PATH = os.path.abspath("data/imagenet_val.npz")


def main():
    if not os.path.exists(MODEL_PATH):
        print(json.dumps({"error": f"Model not found at {MODEL_PATH}"}))
        return

    if not os.path.exists(DATA_PATH):
        print(json.dumps({"error": f"Data not found at {DATA_PATH}"}))
        return

    file_size_bytes = os.path.getsize(MODEL_PATH)
    file_size_kb = round(file_size_bytes / 1024.0, 2)
    file_size_mb = round(file_size_bytes / (1024.0 * 1024.0), 2)

    # Load ONNX session
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Load 4D ImageNet validation dataset
    data = np.load(DATA_PATH)
    X = data["X"].astype(np.float32)  # shape (50, 3, 224, 224)
    y = data["y"]

    # Warmup
    for _ in range(3):
        _ = sess.run(None, {input_name: X[:1]})

    # Benchmark latency
    RUNS = 10
    t0 = time.perf_counter()
    for _ in range(RUNS):
        outputs = sess.run(None, {input_name: X})
    total_time_sec = time.perf_counter() - t0
    latency_ms_per_pass = round((total_time_sec / RUNS) * 1000.0, 3)

    # Accuracy computation
    logits = outputs[0]
    preds = np.argmax(logits, axis=-1)
    accuracy = float(np.mean(preds == y))

    result = {
        "variant": "onnx_fp32",
        "file_size_mb": file_size_mb,
        "file_size_kb": file_size_kb,
        "latency_ms": latency_ms_per_pass,
        "accuracy": round(accuracy, 4),
        "total_benchmark_passes": RUNS,
        "samples_evaluated": len(y)
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
