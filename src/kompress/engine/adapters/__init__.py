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
    return get_adapter(framework).load(artifact_path, framework=framework, **kwargs)


__all__ = ["ModelAdapter", "get_adapter", "load_model", "SklearnAdapter", "PyTorchAdapter"]
