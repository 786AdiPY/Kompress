"""Common interface every framework adapter implements.

The pipeline only ever talks to this interface, so compress/benchmark/serve
stay framework-agnostic. Concrete adapters translate a native model into:
  1. probabilities/predictions (for benchmarking against ONNX variants), and
  2. an ONNX FP32 graph (the universal compression + serving target).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class ModelAdapter(ABC):
    #: framework key from pipeline.yaml (set on the instance in load())
    framework: str = "base"

    def __init__(self, model, framework: str, task: str = "binary_classification",
                 num_classes: int = 2, artifact_path: str | None = None):
        self.model = model
        self.framework = framework
        self.task = task
        self.num_classes = num_classes
        self.artifact_path = artifact_path

    # ── construction ───────────────────────────────────────────────────────────
    @classmethod
    @abstractmethod
    def load(cls, path: str, framework: str, task: str = "binary_classification",
             num_classes: int = 2, **kwargs) -> "ModelAdapter":
        """Load a native model artifact from disk."""

    # ── inference (native reference for the benchmark/gate) ────────────────────
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw model output for the native model.

        binary_classification -> 1-D array of positive-class probabilities
        multiclass            -> 2-D array (n, num_classes) of probabilities
        regression            -> 1-D array of predictions
        """

    # ── export to the universal ONNX FP32 target ───────────────────────────────
    @abstractmethod
    def to_onnx(self, onnx_path: str, n_features: int) -> str:
        """Serialize the model to an ONNX FP32 graph. Returns the path."""

    # ── metadata ───────────────────────────────────────────────────────────────
    @property
    def native_size_kb(self) -> float:
        if self.artifact_path and os.path.exists(self.artifact_path):
            return round(os.path.getsize(self.artifact_path) / 1024, 1)
        return 0.0
