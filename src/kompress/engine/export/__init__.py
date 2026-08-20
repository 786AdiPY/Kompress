"""Export-target registry: ONNX artifact -> device-specific deployable format.

The same target_hardware from the job manifest that picks compressors also picks a
recommended export format here, so compression and deployment stay one concept
end to end. Add a target by importing it and adding it to _TARGETS.
"""
from __future__ import annotations

from .base import ExportTarget
from .coreml_export import CoreMLExport
from .onnx_export import OnnxExport
from .tensorrt_export import TensorRTExport
from .tflite_export import TFLiteExport

_TARGETS: list[ExportTarget] = [
    OnnxExport(),
    TensorRTExport(),
    TFLiteExport(),
    CoreMLExport(),
]
_REGISTRY = {t.format: t for t in _TARGETS}

# Recommended export format per deployment hardware (keys match common/hardware.py).
HARDWARE_FORMAT = {
    "cpu-generic": "onnx",
    "arm-npu": "tflite",
    "sony-imx500": "onnx",   # until a dedicated Sony MCT / IMX500 target is added
    "nvidia-gpu": "tensorrt",
}


def get_exporter(fmt: str) -> ExportTarget:
    if fmt not in _REGISTRY:
        raise KeyError(f"Unknown export format '{fmt}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[fmt]


def list_exporters(hardware: str | None = None) -> list[dict]:
    """All export targets (for the 'Download for device' UI). If hardware is given,
    the format recommended for it is flagged."""
    rec = HARDWARE_FORMAT.get(hardware or "", "onnx")
    out = []
    for t in _TARGETS:
        d = t.describe()
        d["recommended"] = (t.format == rec)
        out.append(d)
    return out


def recommended_format(hardware: str | None) -> str:
    return HARDWARE_FORMAT.get(hardware or "", "onnx")


__all__ = ["ExportTarget", "get_exporter", "list_exporters", "recommended_format"]
