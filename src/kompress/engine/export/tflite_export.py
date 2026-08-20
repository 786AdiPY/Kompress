"""TFLite export for Android / edge-TPU / microcontrollers (TFLite Micro).

Stub: the interface is ready and wired into the registry/UI as 'unavailable' so it
shows up as a device option with a clear enable path. Fill in ONNX -> TF -> TFLite
(via onnx2tf or onnx-tf + tensorflow) to activate — those are heavyweight deps and
need on-device validation, deliberately kept out of the default install."""
from __future__ import annotations

from .base import ExportTarget

_ENABLE = ("TFLite export needs `tensorflow` + `onnx2tf`. Install them and implement "
           "ONNX -> TF SavedModel -> TFLite in export/tflite_export.py to enable.")


class TFLiteExport(ExportTarget):
    format = "tflite"
    label = "TFLite (Android / microcontroller)"
    devices = ["Android", "Edge TPU", "Microcontroller (TFLite Micro)"]
    extension = ".tflite"
    available = False
    unavailable_reason = _ENABLE

    def export(self, onnx_path: str, out_dir: str) -> str:
        raise NotImplementedError(_ENABLE)
