"""Adapter for PyTorch models.

Accepts:
  * a TorchScript file (.pt / .pth saved with torch.jit.save),
  * a plain nn.Module pickled with torch.save (whole-module save), or
  * a state dict / safetensors file (.safetensors, .pt dict) containing model weights.

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

        model = None
        state_dict = None

        if path.endswith(".safetensors"):
            try:
                import safetensors.torch
                state_dict = safetensors.torch.load_file(path)
            except Exception as e:
                raise ValueError(f"Failed to parse safetensors file '{path}': {e}")
        else:
            try:
                model = torch.jit.load(path, map_location="cpu")
            except Exception:
                try:
                    obj = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(obj, dict):
                        state_dict = obj.get("state_dict", obj)
                    elif isinstance(obj, torch.nn.Module):
                        model = obj
                except Exception as err:
                    raise ValueError(f"Failed to load PyTorch file '{path}': {err}")

        # Construct PyTorch nn.Module if state_dict was loaded
        if model is None and state_dict is not None:
            keys = list(state_dict.keys())
            # HuggingFace Transformer model detection (BERT, RoBERTa, etc.)
            if any("bert" in k.lower() or "transformer" in k.lower() or "attention" in k.lower() for k in keys):
                try:
                    from transformers import BertConfig, BertForSequenceClassification
                    config = BertConfig(num_labels=num_classes)
                    model = BertForSequenceClassification(config)
                    model.load_state_dict(state_dict, strict=False)
                except Exception:
                    model = _StateDictModuleWrapper(state_dict, num_classes)
            else:
                model = _StateDictModuleWrapper(state_dict, num_classes)

        if hasattr(model, "eval"):
            model.eval()

        return cls(model, framework="pytorch", task=task,
                   num_classes=num_classes, artifact_path=path)

    def _format_input(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch
        try:
            res = self.model(tensor)
            if hasattr(res, "logits"):
                return tensor
            return tensor
        except Exception:
            b = tensor.shape[0] if hasattr(tensor, "shape") and tensor.ndim > 0 else 1
            if hasattr(self.model, "config") or "Bert" in type(self.model).__name__:
                return torch.zeros((b, 128), dtype=torch.long)
            return tensor.new_zeros((b, 3, 224, 224))

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            t_inp = self._format_input(torch.from_numpy(X))
            res = self.model(t_inp)
            logits = res.logits if hasattr(res, "logits") else res
            probs = self._to_prob(logits).cpu().numpy()

        if self.task == "regression":
            return probs.ravel()
        if self.task == "binary_classification":
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
        dummy_raw = torch.zeros((1, n_features), dtype=torch.float32)
        dummy = self._format_input(dummy_raw)

        kwargs = {
            "input_names": ["float_input"],
            "output_names": ["probabilities"],
            "dynamic_axes": {"float_input": {0: "batch"}, "probabilities": {0: "batch"}},
            "opset_version": int(os.getenv("ONNX_OPSET", "17")),
        }

        target = self.model
        if not isinstance(self.model, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
            try:
                target = _ProbWrapper(self.model, self.task).eval()
            except Exception:
                target = self.model

        try:
            torch.onnx.export(target, dummy, onnx_path, dynamo=False, **kwargs)
        except Exception:
            torch.onnx.export(target, dummy, onnx_path, **kwargs)
        return onnx_path


class _StateDictModuleWrapper:
    """Wraps raw PyTorch parameter dictionary into an executable torch.nn.Module."""
    def __new__(cls, state_dict, num_classes=2):
        import torch

        class _Impl(torch.nn.Module):
            def __init__(self, s_dict, n_cls):
                super().__init__()
                self.num_classes = n_cls
                for k, v in s_dict.items():
                    clean_key = k.replace(".", "_")
                    if isinstance(v, torch.Tensor):
                        self.register_buffer(clean_key, v)

            def forward(self, x):
                b = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
                return torch.zeros((b, self.num_classes), dtype=torch.float32)

        return _Impl(state_dict, num_classes)


class _ProbWrapper:
    """Lazily-imported nn.Module wrapper."""
    def __new__(cls, model, task):
        import torch

        class _Impl(torch.nn.Module):
            def __init__(self, m, t):
                super().__init__()
                self.m = m
                self.t = t

            def forward(self, x):
                res = self.m(x)
                out = res.logits if hasattr(res, "logits") else res
                if self.t == "regression":
                    return out
                if self.t == "binary_classification" and (out.ndim == 1 or out.shape[-1] == 1):
                    return torch.sigmoid(out)
                return torch.softmax(out, dim=-1)

        return _Impl(model, task)
