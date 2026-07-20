"""Generic multi-model compression stage.

For each model in pipeline.yaml: load it via its framework adapter, export ONNX
FP32, run every configured compressor, and write a per-model manifest
(artifacts/<name>/variants.json). Finally writes artifacts/models.json — the
index every downstream stage iterates over. Fully framework-agnostic.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config, feature_names          # noqa: E402
from adapters import load_model                          # noqa: E402
from compressors import get_compressor                   # noqa: E402
from compressors.base import Variant                     # noqa: E402


def load_calibration(spec, n: int = 1000):
    """Representative rows from the model's own test set for static/TRT calib."""
    if not os.path.exists(spec.test_csv):
        return None
    df = pd.read_csv(spec.test_csv)
    cols = [c for c in feature_names(spec.features) if c in df.columns]
    if len(cols) != len(spec.features):
        return None
    return df[cols].head(n).values.astype(np.float32)


def compress_model(spec) -> dict:
    names = feature_names(spec.features)
    n_features = len(names)
    os.makedirs(spec.out_dir, exist_ok=True)

    print(f"\n=== Compressing '{spec.name}'  framework={spec.framework}  "
          f"task={spec.task}  features={n_features} ===")

    if not os.path.exists(spec.artifact):
        raise FileNotFoundError(
            f"[{spec.name}] artifact not found: {spec.artifact}. "
            f"Train it or drop your trained model there.")

    adapter = load_model(spec.framework, spec.artifact,
                         task=spec.task, num_classes=spec.num_classes)
    adapter.to_onnx(spec.onnx_fp32_path, n_features)
    print(f"ONNX FP32 exported -> {spec.onnx_fp32_path} "
          f"({Variant.size_of(spec.onnx_fp32_path):.1f} KB)")

    native = Variant(name=f"native_{spec.framework}", kind="native",
                     path=spec.artifact, size_kb=adapter.native_size_kb,
                     note="uncompressed reference")

    calib = load_calibration(spec)
    variants: list[Variant] = []
    for method in spec.compression_methods:
        print(f"--- compressor: {method} ---")
        try:
            v = get_compressor(method).compress(spec.onnx_fp32_path, spec.out_dir, calib_data=calib)
        except Exception as e:
            print(f"[{method}] errored: {e}; skipping.")
            v = None
        if v is not None:
            variants.append(v)

    manifest = {
        "name":        spec.name,
        "framework":   spec.framework,
        "task":        spec.task,
        "num_classes": spec.num_classes,
        "features":    spec.features,
        "target":      spec.target,
        "test_csv":    spec.test_csv,
        "native":      native.to_dict(),
        "variants":    [v.to_dict() for v in variants],
    }
    with open(spec.manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest -> {spec.manifest_path}  variants: {[v.name for v in variants]}")

    return {"name": spec.name, "framework": spec.framework, "task": spec.task,
            "manifest": spec.manifest_path}


def main():
    cfg = load_config()
    os.makedirs(os.getenv("ARTIFACTS_DIR", "artifacts"), exist_ok=True)

    index_entries = []
    for spec in cfg.models:
        try:
            index_entries.append(compress_model(spec))
        except Exception as e:
            print(f"[{spec.name}] compression failed: {e}")

    index = {"project": cfg.project, "models": index_entries}
    with open(cfg.models_index, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nModels index -> {cfg.models_index}  "
          f"({len(index_entries)}/{len(cfg.models)} models compressed)")

    if not index_entries:
        sys.exit(1)


if __name__ == "__main__":
    main()
