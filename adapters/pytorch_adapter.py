"""Adapter for PyTorch models.

Accepts either:
  * a TorchScript file (.pt / .pth saved with torch.jit.save), or
  * a plain nn.Module pickled with torch.save (whole-module save).

Exports to ONNX with torch.onnx.export. For classification the module is
expected to output raw logits of shape (n, num_classes); the adapter applies
softmax/sigmoid to produce probabilities so it matches the ONNX graph, which
we also make emit probabilities.
"""
from __future__ import annotations

import os

import numpy as np

from .base import ModelAdapter


class PyTorchAdapter(ModelAdapter):
    @classmethod
    def load(cls, path, framework="pytorch", task="binary_classification", num_classes=2, **kwargs):
        import torch

        try:
            model = torch.jit.load(path, map_location="cpu")
        except Exception:
            model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        return cls(model, framework="pytorch", task=task,
                   num_classes=num_classes, artifact_path=path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X))
            probs = self._to_prob(logits).cpu().numpy()

        if self.task == "regression":
            return probs.ravel()
        if self.task == "binary_classification":
            # (n, 2) -> positive class; (n, 1)/(n,) -> sigmoid output
            return probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.ravel()
        return probs

    def _to_prob(self, logits):
        import torch

        if self.task == "regression":
            return logits
        if self.task == "binary_classification" and (logits.ndim == 1 or logits.shape[-1] == 1):
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def to_onnx(self, onnx_path: str, n_features: int) -> str:
        import torch

        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
        dummy = torch.zeros((1, n_features), dtype=torch.float32)

        # Wrap so the exported graph emits probabilities (parity with predict()).
        wrapper = _ProbWrapper(self.model, self.task).eval()

        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["float_input"],
            output_names=["probabilities"],
            dynamic_axes={"float_input": {0: "batch"}, "probabilities": {0: "batch"}},
            opset_version=int(os.getenv("ONNX_OPSET", "17")),
        )
        return onnx_path


class _ProbWrapper:
    """Lazily-imported nn.Module wrapper (defined via __init_subclass__-free
    factory to avoid importing torch at module import time)."""

    def __new__(cls, model, task):
        import torch

        class _Impl(torch.nn.Module):
            def __init__(self, m, t):
                super().__init__()
                self.m = m
                self.t = t

            def forward(self, x):
                out = self.m(x)
                if self.t == "regression":
                    return out
                if self.t == "binary_classification" and (out.ndim == 1 or out.shape[-1] == 1):
                    return torch.sigmoid(out)
                return torch.softmax(out, dim=-1)

        return _Impl(model, task)
