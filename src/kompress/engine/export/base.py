"""Export-target interface: convert the universal ONNX artifact into the format a
specific deployment device actually runs.

This is the "where it goes out" layer, mirroring compressors/ ("how it's made
smaller") and adapters/ ("what comes in"). ONNX is the hub; every device target
is a converter from ONNX. Add a target by dropping a module in and listing it in
export/__init__.py — nothing else changes.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class ExportTarget(ABC):
    #: short id used on the wire, e.g. onnx | tensorrt | tflite | coreml
    format: str = "base"
    #: human label for the UI
    label: str = "Base"
    #: devices this format serves (shown as "Download for ...")
    devices: list[str] = []
    #: file extension of the produced artifact
    extension: str = ".bin"
    #: False when the converter's deps/hardware are missing on this host — the
    #: interface still exists so the UI can show it as "unavailable" with a reason.
    available: bool = True
    #: why it's unavailable / how to enable it (shown to the user)
    unavailable_reason: str = ""

    @abstractmethod
    def export(self, onnx_path: str, out_dir: str) -> str:
        """Produce the deployable artifact from an ONNX model; return its path.

        Raise RuntimeError/NotImplementedError with an actionable message when the
        conversion can't run here (missing runtime, no GPU, unimplemented stub)."""

    def runtime_hint(self) -> str:
        """One-line 'how to run it on the device' note, shown next to the download."""
        return ""

    def describe(self) -> dict:
        return {
            "format": self.format,
            "label": self.label,
            "devices": self.devices,
            "extension": self.extension,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "runtime_hint": self.runtime_hint(),
        }
