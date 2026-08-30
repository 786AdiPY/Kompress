"""Feature-schema helpers: one feature list drives the ONNX input shape,
the serve request model, and array conversion — no more duplicated FEATURE_COLS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, create_model, field_validator

_DTYPE_MAP = {"float": float, "int": int, "double": float, "long": int}


@dataclass
class FeatureSpec:
    name: str
    dtype: str = "float"
    choices: list | None = None

    @property
    def py_type(self):
        return _DTYPE_MAP.get(self.dtype, float)


def parse_features(features: list[dict[str, Any]]) -> list[FeatureSpec]:
    return [
        FeatureSpec(name=f["name"], dtype=f.get("dtype", "float"), choices=f.get("choices"))
        for f in features
    ]


def feature_names(features: list[dict[str, Any]]) -> list[str]:
    return [f["name"] for f in features]


def build_request_model(features: list[dict[str, Any]], name: str = "PredictRequest"):
    """Dynamically build a Pydantic request model from the feature spec.

    Replaces the hardcoded ChurnRequest so serve/ works for any model.
    Validators are passed at class-creation time (required for Pydantic v2).
    """
    specs = parse_features(features)
    fields: dict[str, tuple] = {s.name: (s.py_type, ...) for s in specs}

    validators: dict[str, Any] = {}
    for s in specs:
        if not s.choices:
            continue
        allowed = sorted(set(s.choices))
        fname = s.name

        def _make_validator(field_name, allowed_set):
            def _validate(cls, v):
                if v not in allowed_set:
                    raise ValueError(f"{field_name} must be one of {allowed_set}")
                return v
            return classmethod(_validate)

        validators[f"validate_{fname}"] = field_validator(fname)(_make_validator(fname, allowed))

    return create_model(name, __validators__=validators or None, **fields)


def records_to_array(records, features: list[dict[str, Any]]) -> np.ndarray:
    """Convert a dict (or list of dicts) into a float32 model-input array."""
    names = feature_names(features)
    if isinstance(records, dict):
        records = [records]
    rows = [[_get(r, n) for n in names] for r in records]
    return np.asarray(rows, dtype=np.float32)


def _get(record, name):
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)
