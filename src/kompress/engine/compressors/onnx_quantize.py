"""ONNX Runtime quantization — real INT8 compression on CPU, no GPU required.

  * OnnxFp32          — passthrough baseline (the exported graph).
  * OnnxInt8Dynamic   — dynamic quantization; weights INT8, activations quantized
                        at runtime. Universal, needs no calibration data.  ⭐
  * OnnxInt8Static    — static quantization using calibration samples; smaller
                        and faster still, but needs representative data.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import Compressor, Variant


class OnnxFp32(Compressor):
    name = "onnx_fp32"

    def compress(self, fp32_onnx_path, out_dir, *, calib_data=None) -> Optional[Variant]:
        # Baseline: the FP32 graph itself is the variant.
        return Variant(
            name=self.name, kind="onnx", path=fp32_onnx_path,
            size_kb=Variant.size_of(fp32_onnx_path), note="baseline (no compression)",
        )


class OnnxInt8Dynamic(Compressor):
    name = "onnx_int8_dynamic"

    def compress(self, fp32_onnx_path, out_dir, *, calib_data=None) -> Optional[Variant]:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except Exception as e:  # pragma: no cover
            print(f"[{self.name}] onnxruntime.quantization unavailable: {e}")
            return None

        out_path = os.path.join(out_dir, "model_int8_dynamic.onnx")
        try:
            quantize_dynamic(
                fp32_onnx_path, out_path, weight_type=QuantType.QInt8,
            )
        except Exception as e:
            print(f"[{self.name}] dynamic quantization failed ({e}); skipping.")
            return None

        print(f"[{self.name}] INT8 ONNX saved -> {out_path} "
              f"({Variant.size_of(out_path):.1f} KB)")
        return Variant(name=self.name, kind="onnx", path=out_path,
                       size_kb=Variant.size_of(out_path))


class OnnxInt8Static(Compressor):
    name = "onnx_int8_static"

    def compress(self, fp32_onnx_path, out_dir, *, calib_data=None) -> Optional[Variant]:
        if calib_data is None or len(calib_data) == 0:
            print(f"[{self.name}] no calibration data provided; skipping.")
            return None
        try:
            from onnxruntime.quantization import (
                quantize_static, QuantType, CalibrationDataReader,
            )
            import onnxruntime as ort
        except Exception as e:  # pragma: no cover
            print(f"[{self.name}] onnxruntime.quantization unavailable: {e}")
            return None

        input_name = ort.InferenceSession(
            fp32_onnx_path, providers=["CPUExecutionProvider"]
        ).get_inputs()[0].name

        class _Reader(CalibrationDataReader):
            def __init__(self, data):
                self._it = iter(
                    {input_name: data[i:i + 1].astype(np.float32)}
                    for i in range(len(data))
                )

            def get_next(self):
                return next(self._it, None)

        out_path = os.path.join(out_dir, "model_int8_static.onnx")
        try:
            quantize_static(
                fp32_onnx_path, out_path, _Reader(np.asarray(calib_data)),
                weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
            )
        except Exception as e:
            print(f"[{self.name}] static quantization failed ({e}); skipping.")
            return None

        print(f"[{self.name}] static INT8 ONNX saved -> {out_path} "
              f"({Variant.size_of(out_path):.1f} KB)")
        return Variant(name=self.name, kind="onnx", path=out_path,
                       size_kb=Variant.size_of(out_path))
