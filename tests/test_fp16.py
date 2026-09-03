"""Tests for FP16 and Adaptive Mixed-FP16 precision conversion."""

import os
import tempfile
import numpy as np
import pytest
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
import onnx
import onnxruntime as ort

from kompress.engine.compressors.onnx_fp16 import (
    OnnxFp16,
    OnnxFp16Mixed,
    OnnxFp16Adaptive,
    get_sensitive_node_names,
)


def create_sample_onnx_model(tmp_dir: str) -> str:
    """Helper to generate a simple FP32 ONNX model."""
    X = np.random.randn(100, 4).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)

    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onnx_model = skl2onnx.convert_sklearn(clf, initial_types=initial_type)
    onnx_path = os.path.join(tmp_dir, "model_fp32.onnx")
    onnx.save(onnx_model, onnx_path)
    return onnx_path


def test_sensitive_nodes_finder():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = create_sample_onnx_model(tmp)
        model = onnx.load(model_path)
        sensitive = get_sensitive_node_names(model)
        assert isinstance(sensitive, list)
        assert len(sensitive) >= 0


def test_onnx_fp16_full_compression():
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = create_sample_onnx_model(tmp)
        compressor = OnnxFp16()
        variant = compressor.compress(fp32_path, tmp)

        assert variant is not None
        assert variant.name == "onnx_fp16"
        assert os.path.exists(variant.path)
        assert variant.size_kb > 0

        # Verify ONNX Runtime session loads and runs
        sess = ort.InferenceSession(variant.path, providers=["CPUExecutionProvider"])
        test_in = np.random.randn(5, 4).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: test_in})
        assert len(out) > 0


def test_onnx_fp16_mixed_compression():
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = create_sample_onnx_model(tmp)
        compressor = OnnxFp16Mixed()
        variant = compressor.compress(fp32_path, tmp)

        assert variant is not None
        assert variant.name == "onnx_fp16_mixed"
        assert os.path.exists(variant.path)
        assert "mixed fp16" in variant.note.lower()

        # Verify execution
        sess = ort.InferenceSession(variant.path, providers=["CPUExecutionProvider"])
        test_in = np.random.randn(5, 4).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: test_in})
        assert len(out) > 0


def test_onnx_fp16_adaptive_controller():
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = create_sample_onnx_model(tmp)
        compressor = OnnxFp16Adaptive()

        calib_data = np.random.randn(50, 4).astype(np.float32)
        variant = compressor.compress(fp32_path, tmp, calib_data=calib_data)

        assert variant is not None
        assert variant.name == "onnx_fp16_adaptive"
        assert os.path.exists(variant.path)


if __name__ == "__main__":
    pytest.main([__file__])
