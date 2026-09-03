"""ONNX Float16 & Adaptive Mixed-Precision Compression.

Follows precision sensitivity concept:
  1. Full FP16 conversion (`OnnxFp16`).
  2. Mixed FP16 conversion (`OnnxFp16Mixed`) keeping sensitive output/classifier layers in FP32.
  3. Adaptive FP16 Controller (`OnnxFp16Adaptive`) benchmarks full FP16 accuracy drop;
     if degradation exceeds allowed threshold, falls back to Mixed FP16.
"""
from __future__ import annotations

import os
from typing import Optional, List, Set
import numpy as np
import onnx

from .base import Compressor, Variant


def get_sensitive_node_names(model: onnx.ModelProto) -> List[str]:
    """Find sensitive graph nodes (output producers, softmax, classifiers) to preserve in FP32."""
    output_names = {out.name for out in model.graph.output}
    sensitive_nodes: Set[str] = set()

    sensitive_op_types = {"Softmax", "LogSoftmax", "ArgMax", "Exp", "Sigmoid", "BatchNormalization"}

    for node in model.graph.node:
        # 1. Output-producing tail nodes
        if any(out_name in output_names for out_name in node.output):
            if node.name:
                sensitive_nodes.add(node.name)
        # 2. Sensitive mathematical op types
        if node.op_type in sensitive_op_types:
            if node.name:
                sensitive_nodes.add(node.name)

    return list(sensitive_nodes)


class OnnxFp16(Compressor):
    name = "onnx_fp16"

    def compress(self, fp32_onnx_path: str, out_dir: str, *, calib_data: Optional[np.ndarray] = None) -> Optional[Variant]:
        try:
            from onnxconverter_common import float16
        except Exception as e:  # pragma: no cover
            print(f"[{self.name}] onnxconverter_common unavailable: {e}")
            return None

        out_path = os.path.join(out_dir, "model_fp16.onnx")
        try:
            model = onnx.load(fp32_onnx_path)
            fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)
            onnx.save(fp16_model, out_path)
        except Exception as e:
            print(f"[{self.name}] FP16 conversion failed ({e}); skipping.")
            return None

        print(f"[{self.name}] Full FP16 ONNX saved -> {out_path} ({Variant.size_of(out_path):.1f} KB)")
        return Variant(
            name=self.name,
            kind="onnx",
            path=out_path,
            size_kb=Variant.size_of(out_path),
            note="full FP16",
        )


class OnnxFp16Mixed(Compressor):
    name = "onnx_fp16_mixed"

    def compress(self, fp32_onnx_path: str, out_dir: str, *, calib_data: Optional[np.ndarray] = None) -> Optional[Variant]:
        try:
            from onnxconverter_common import float16
        except Exception as e:  # pragma: no cover
            print(f"[{self.name}] onnxconverter_common unavailable: {e}")
            return None

        out_path = os.path.join(out_dir, "model_fp16_mixed.onnx")
        try:
            model = onnx.load(fp32_onnx_path)
            block_list = get_sensitive_node_names(model)
            mixed_model = float16.convert_float_to_float16(
                model,
                keep_io_types=True,
                node_block_list=block_list if block_list else None,
            )
            onnx.save(mixed_model, out_path)
        except Exception as e:
            print(f"[{self.name}] Mixed FP16 conversion failed ({e}); skipping.")
            return None

        print(f"[{self.name}] Mixed FP16 ONNX saved -> {out_path} ({Variant.size_of(out_path):.1f} KB)")
        return Variant(
            name=self.name,
            kind="onnx",
            path=out_path,
            size_kb=Variant.size_of(out_path),
            note="mixed FP16 (sensitive layers preserved in FP32)",
        )


class OnnxFp16Adaptive(Compressor):
    name = "onnx_fp16_adaptive"

    def compress(self, fp32_onnx_path: str, out_dir: str, *, calib_data: Optional[np.ndarray] = None) -> Optional[Variant]:
        """Tries Full FP16, evaluates accuracy/output deviation on calib_data.

        If deviation exceeds tolerance, falls back to Mixed FP16.
        """
        full_fp16_comp = OnnxFp16()
        mixed_fp16_comp = OnnxFp16Mixed()

        full_variant = full_fp16_comp.compress(fp32_onnx_path, out_dir, calib_data=calib_data)
        if full_variant is None:
            return None

        # If no calibration/eval data provided, default to full FP16
        if calib_data is None or len(calib_data) == 0:
            return full_variant

        # Compare outputs between FP32 baseline and Full FP16 on calib_data
        try:
            import onnxruntime as ort

            session_fp32 = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
            session_fp16 = ort.InferenceSession(full_variant.path, providers=["CPUExecutionProvider"])

            inputs_32 = [inp.name for inp in session_fp32.get_inputs()]
            inputs_16 = [inp.name for inp in session_fp16.get_inputs()]

            if isinstance(calib_data, dict):
                feed_32 = {k: v[: min(len(v), 100)] for k, v in calib_data.items() if k in inputs_32}
                feed_16 = {k: v[: min(len(v), 100)] for k, v in calib_data.items() if k in inputs_16}
            else:
                test_input = calib_data[: min(len(calib_data), 100)]
                feed_32 = {inputs_32[0]: test_input.astype(np.float32)}
                feed_16 = {inputs_16[0]: test_input.astype(np.float32)}

            out_32 = session_fp32.run(None, feed_32)[0]
            out_16 = session_fp16.run(None, feed_16)[0]

            # Compute Max Absolute Difference or Mean Absolute Error
            mae = float(np.mean(np.abs(out_32 - out_16)))
            max_diff = float(np.max(np.abs(out_32 - out_16)))

            # Threshold for switching to mixed precision (e.g. MAE > 1e-2 or MaxDiff > 0.05)
            degraded = (mae > 0.01) or (max_diff > 0.05)
            print(f"[{self.name}] Full FP16 MAE vs FP32: {mae:.5f}, Max Diff: {max_diff:.5f} -> Degraded: {degraded}")

            if degraded:
                print(f"[{self.name}] Accuracy degradation detected! Falling back to Mixed FP16 (preserving sensitive layers in FP32).")
                mixed_variant = mixed_fp16_comp.compress(fp32_onnx_path, out_dir, calib_data=calib_data)
                if mixed_variant:
                    mixed_variant.name = self.name
                    mixed_variant.note = f"adaptive fallback to mixed FP16 (MAE={mae:.4f})"
                    return mixed_variant

        except Exception as e:
            print(f"[{self.name}] Adaptive precision benchmark error: {e}; retaining full FP16.")

        full_variant.name = self.name
        full_variant.note = "adaptive selected full FP16"
        return full_variant
