"""Compressor registry. Each compressor turns an ONNX FP32 graph into a
compressed variant (or is skipped if its runtime is unavailable).
"""
from __future__ import annotations

from .base import Compressor, Variant
from .onnx_quantize import OnnxFp32, OnnxInt8Dynamic, OnnxInt8Static
from .trt_int8 import TrtInt8

_REGISTRY: dict[str, Compressor] = {
    "onnx_fp32":         OnnxFp32(),
    "onnx_int8_dynamic": OnnxInt8Dynamic(),
    "onnx_int8_static":  OnnxInt8Static(),
    "trt_int8":          TrtInt8(),
}


def get_compressor(name: str) -> Compressor:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown compressor '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = ["Compressor", "Variant", "get_compressor"]
