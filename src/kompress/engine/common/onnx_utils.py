"""Uniform ONNX-Runtime inference helpers.

Different exporters emit different output layouts:
  * onnxmltools XGBoost/LightGBM -> [label, list-of-dicts {0:p0, 1:p1}]
  * skl2onnx (zipmap=False)      -> [label, ndarray (n, C)]
  * torch.onnx (our wrapper)     -> [ndarray (n, C)] probabilities
This module hides those differences behind extract_proba().
"""
from __future__ import annotations

import numpy as np


def make_session(onnx_path: str):
    import onnxruntime as ort
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def run(sess, X: np.ndarray):
    inputs = sess.get_inputs()
    feed = {}
    for inp_meta in inputs:
        expected_type = str(inp_meta.type or "tensor(float)")
        if "int64" in expected_type:
            dtype = np.int64
        elif "int32" in expected_type:
            dtype = np.int32
        else:
            dtype = np.float32

        arr = np.asarray(X, dtype=dtype)
        expected_shape = inp_meta.shape

        if len(expected_shape) == 4 and arr.ndim == 2:
            b = arr.shape[0]
            c = expected_shape[1] if isinstance(expected_shape[1], int) else 3
            h = expected_shape[2] if isinstance(expected_shape[2], int) else 224
            w = expected_shape[3] if isinstance(expected_shape[3], int) else 224
            arr = np.zeros((b, c, h, w), dtype=dtype)
        elif len(expected_shape) == 2 and arr.ndim == 2:
            req_dim1 = expected_shape[1]
            if isinstance(req_dim1, int) and req_dim1 != arr.shape[1]:
                b = arr.shape[0]
                new_arr = np.zeros((b, req_dim1), dtype=dtype)
                col_copy = min(arr.shape[1], req_dim1)
                new_arr[:, :col_copy] = arr[:, :col_copy]
                arr = new_arr

        feed[inp_meta.name] = arr

    return sess.run(None, feed)


def extract_proba(outputs, task: str = "binary_classification") -> np.ndarray:
    """Normalize raw ONNX outputs into probabilities.

    binary -> 1-D positive-class probs; multiclass -> (n, C); regression -> 1-D.
    """
    if task == "regression":
        arr = np.asarray(outputs[0])
        return arr.ravel()

    # Find the probability payload: prefer a 2-D float tensor or a list-of-dicts.
    prob = _find_prob_payload(outputs)

    if isinstance(prob, list):  # list of {class: p} dicts
        classes = sorted(prob[0].keys())
        mat = np.array([[d[c] for c in classes] for d in prob], dtype=np.float32)
    else:
        mat = np.asarray(prob, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)

    if task == "binary_classification":
        return mat[:, 1] if mat.shape[1] >= 2 else mat.ravel()
    return mat


def _find_prob_payload(outputs):
    # Scan outputs for the most probability-like element.
    for o in outputs:
        if isinstance(o, list) and o and isinstance(o[0], dict):
            return o
    for o in outputs:
        arr = np.asarray(o)
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
            return arr
    # Fallback: last output.
    return outputs[-1]
