"""Shared, framework-agnostic core for the compression pipeline."""
from .config import PipelineConfig, load_config
from .schema import FeatureSpec, feature_names, build_request_model, records_to_array

__all__ = [
    "PipelineConfig",
    "load_config",
    "FeatureSpec",
    "feature_names",
    "build_request_model",
    "records_to_array",
]
