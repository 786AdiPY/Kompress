"""Framework adapters: load any model and export it to ONNX FP32.

Register new frameworks by adding an adapter class and listing it in _REGISTRY.
"""
from __future__ import annotations

from .base import ModelAdapter
from .sklearn_adapter import SklearnAdapter
from .pytorch_adapter import PyTorchAdapter

# framework name (as used in pipeline.yaml) -> adapter class
_REGISTRY: dict[str, type[ModelAdapter]] = {
    "xgboost":  SklearnAdapter,
    "lightgbm": SklearnAdapter,
    "sklearn":  SklearnAdapter,
    "pytorch":  PyTorchAdapter,
}


def get_adapter(framework: str) -> type[ModelAdapter]:
    if framework not in _REGISTRY:
        raise ValueError(
            f"Unsupported framework '{framework}'. "
            f"Supported: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[framework]


def load_model(framework: str, artifact_path: str, **kwargs) -> ModelAdapter:
    # Auto-detect PyTorch models (.pt / .pth) when framework was mismatched (e.g. xgboost selected for model.pt)
    if artifact_path.lower().endswith((".pt", ".pth")) and framework in ("xgboost", "sklearn", "lightgbm"):
        try:
            return PyTorchAdapter.load(artifact_path, framework="pytorch", **kwargs)
        except Exception:
            pass

    try:
        return get_adapter(framework).load(artifact_path, framework=framework, **kwargs)
    except Exception as err:
        if artifact_path.lower().endswith((".pt", ".pth")):
            try:
                return PyTorchAdapter.load(artifact_path, framework="pytorch", **kwargs)
            except Exception:
                pass
        raise ValueError(
            f"Failed to load model file '{artifact_path}' with framework '{framework}'. "
            f"Please ensure the selected framework matches the model format. Details: {err}"
        ) from err


__all__ = ["ModelAdapter", "get_adapter", "load_model", "SklearnAdapter", "PyTorchAdapter"]
