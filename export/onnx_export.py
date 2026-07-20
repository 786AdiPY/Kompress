"""ONNX export — the universal target. ONNX Runtime runs on x86/ARM servers,
Raspberry Pi, and mobile (ONNX Runtime Mobile), so the compressed ONNX is already
deployable as-is; this exporter just hands it over with a runtime hint."""
from __future__ import annotations

import os
import shutil

from .base import ExportTarget


class OnnxExport(ExportTarget):
    format = "onnx"
    label = "ONNX (ONNX Runtime)"
    devices = ["CPU server", "Raspberry Pi / ARM", "Mobile (ORT Mobile)", "Any"]
    extension = ".onnx"
    available = True

    def export(self, onnx_path: str, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, "model.onnx")
        shutil.copy(onnx_path, dst)
        return dst

    def runtime_hint(self) -> str:
        return ("import onnxruntime as ort; "
                "sess = ort.InferenceSession('model.onnx'); "
                "sess.run(None, {sess.get_inputs()[0].name: x})")
