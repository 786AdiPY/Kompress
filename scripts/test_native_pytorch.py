#!/usr/bin/env python3
"""Standalone empirical benchmark script for native PyTorch ResNet-50 model."""
import json
import os
import time
import numpy as np
import torch

MODEL_PATH = os.path.abspath("artifacts/resnet50/model.pt")
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

    # Load model
    model = torch.jit.load(MODEL_PATH, map_location="cpu").eval()

    # Load 4D ImageNet validation dataset
    data = np.load(DATA_PATH)
    X = data["X"]  # shape (50, 3, 224, 224)
    y = data["y"]  # ground truth class IDs

    # Warmup
    X_tensor = torch.from_numpy(X)
    with torch.no_grad():
        for _ in range(3):
            _ = model(X_tensor[:1])

    # Benchmark latency
    RUNS = 10
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            logits = model(X_tensor)
    total_time_sec = time.perf_counter() - t0
    latency_ms_per_pass = round((total_time_sec / RUNS) * 1000.0, 3)

    # Accuracy computation
    preds = logits.argmax(dim=-1).numpy()
    accuracy = float(np.mean(preds == y))

    result = {
        "variant": "native_pytorch",
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
