"""Load and resolve pipeline.yaml into typed config objects.

Supports MULTIPLE models: `pipeline.yaml` has a `models:` list and every stage
loops over it, compressing each model independently. A legacy single `model:`
block (plus top-level features/target/data) is still accepted for back-compat.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.getenv("PIPELINE_CONFIG", "pipeline.yaml")
MODELS_INDEX = os.getenv("MODELS_INDEX", "artifacts/models.json")


@dataclass
class ModelSpec:
    """One self-contained model to compress: its framework, task, artifact,
    input schema, and its own train/test data."""
    name: str
    framework: str
    task: str
    artifact: str
    num_classes: int
    features: list[dict[str, Any]]
    target: str
    train_csv: str
    test_csv: str
    trainable: bool
    compression_methods: list[str]
    gate: dict[str, Any]

    # ── per-model artifact layout: artifacts/<name>/... ────────────────────────
    @property
    def out_dir(self) -> str:
        return os.path.join(os.getenv("ARTIFACTS_DIR", "artifacts"), self.name)

    @property
    def onnx_fp32_path(self) -> str:
        return os.path.join(self.out_dir, "model_fp32.onnx")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.out_dir, "variants.json")

    @property
    def is_classification(self) -> bool:
        return self.task in ("binary_classification", "multiclass_classification")


@dataclass
class PipelineConfig:
    project: str
    models: list[ModelSpec]
    deploy: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def models_index(self) -> str:
        return MODELS_INDEX


def _feature_names(model_dict, defaults):
    return model_dict.get("features", defaults.get("features", []))


def _build_spec(m: dict, defaults: dict) -> ModelSpec:
    """Build one ModelSpec, inheriting global defaults where a field is absent."""
    data = {**defaults.get("data", {}), **m.get("data", {})}
    compression = {**defaults.get("compression", {}), **m.get("compression", {})}
    gate = {**defaults.get("gate", {}), **m.get("gate", {})}
    framework = m.get("framework", defaults.get("framework", "xgboost"))

    return ModelSpec(
        name=m.get("name", framework),
        framework=framework,
        task=m.get("task", defaults.get("task", "binary_classification")),
        artifact=m.get("artifact", defaults.get("artifact", f"artifacts/{m.get('name','model')}/model.pkl")),
        num_classes=int(m.get("num_classes", defaults.get("num_classes", 2))),
        features=m.get("features", defaults.get("features", [])),
        target=m.get("target", defaults.get("target", "target")),
        train_csv=data.get("train", "data/train.csv"),
        test_csv=data.get("test", "data/test.csv"),
        trainable=bool(m.get("trainable", False)),
        compression_methods=list(compression.get("methods", ["onnx_fp32"])),
        gate=gate,
    )


def load_config(path: str | None = None) -> PipelineConfig:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Global defaults every model inherits unless it overrides.
    defaults = {
        "features": data.get("features", []),
        "target": data.get("target", "target"),
        "data": data.get("data", {}),
        "compression": data.get("compression", {}),
        "gate": data.get("gate", {}),
        **(data.get("model") or {}),  # legacy single-model fields as defaults
    }

    if data.get("models"):
        specs = [_build_spec(m, defaults) for m in data["models"]]
    else:
        # Legacy: one `model:` block + top-level features/target/data.
        legacy = dict(data.get("model") or {})
        legacy.setdefault("name", legacy.get("framework", "model"))
        specs = [_build_spec(legacy, defaults)]

    return PipelineConfig(
        project=data.get("project", "model"),
        models=specs,
        deploy=data.get("deploy", {}),
        raw=data,
    )
