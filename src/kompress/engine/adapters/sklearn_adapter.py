"""Adapter for classic-ML models: XGBoost, LightGBM and scikit-learn.

All three pickle to a single .pkl and export to ONNX via onnxmltools
(tree boosters) or skl2onnx (generic sklearn estimators).
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from .base import ModelAdapter


class SklearnAdapter(ModelAdapter):
    @classmethod
    def load(cls, path, framework, task="binary_classification", num_classes=2, **kwargs):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return cls(model, framework=framework, task=task,
                   num_classes=num_classes, artifact_path=path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.task == "regression":
            return np.asarray(self.model.predict(X)).ravel()

        proba = self.model.predict_proba(X)
        proba = np.asarray(proba)
        if self.task == "binary_classification":
            return proba[:, 1]
        return proba  # multiclass -> full matrix

    def to_onnx(self, onnx_path: str, n_features: int) -> str:
        onnx_model = self._convert(n_features)
        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return onnx_path

    # ── conversion dispatch ────────────────────────────────────────────────────
    def _convert(self, n_features: int):
        model_cls = type(self.model).__name__.lower()

        # onnxmltools (XGBoost/LightGBM) and skl2onnx each require their OWN
        # FloatTensorType class — they are not interchangeable.
        if "xgb" in model_cls or self.framework == "xgboost":
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            return convert_xgboost(self.model, initial_types=initial_type)

        if "lgb" in model_cls or "lightgbm" in model_cls or self.framework == "lightgbm":
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            return convert_lightgbm(self.model, initial_types=initial_type)

        # Generic scikit-learn estimator/pipeline.
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        options = None
        if self.task in ("binary_classification", "multiclass_classification"):
            # Emit clean probability tensors (not ZipMap dicts) for easy parsing.
            options = {id(self.model): {"zipmap": False}}
        return convert_sklearn(self.model, initial_types=initial_type, options=options)
