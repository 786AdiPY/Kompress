"""CoreML export for iOS / Apple Neural Engine.

Stub: interface ready, shown as 'unavailable' with an enable path. Fill in
ONNX -> CoreML (via coremltools) to activate — heavyweight dep + on-device
validation, kept out of the default install."""
from __future__ import annotations

from .base import ExportTarget

_ENABLE = ("CoreML export needs `coremltools`. Install it and implement "
           "ONNX -> CoreML in export/coreml_export.py to enable.")


class CoreMLExport(ExportTarget):
    format = "coreml"
    label = "CoreML (iOS / Apple Neural Engine)"
    devices = ["iPhone / iPad", "Apple Neural Engine"]
    extension = ".mlpackage"
    available = False
    unavailable_reason = _ENABLE

    def export(self, onnx_path: str, out_dir: str) -> str:
        raise NotImplementedError(_ENABLE)
