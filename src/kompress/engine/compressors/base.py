"""Compressor interface + the Variant descriptor the rest of the pipeline uses."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class Variant:
    """A servable/benchmarkable model artifact produced by a compressor."""
    name: str            # e.g. "onnx_int8_dynamic"
    kind: str            # "onnx" | "trt" | "native"
    path: str            # file on disk to load
    size_kb: float = 0.0
    note: str = ""       # e.g. "fallback: no GPU"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def size_of(path: str) -> float:
        return round(os.path.getsize(path) / 1024, 1) if os.path.exists(path) else 0.0


class Compressor(ABC):
    name: str = "base"

    @abstractmethod
    def compress(
        self,
        fp32_onnx_path: str,
        out_dir: str,
        *,
        calib_data: Optional[np.ndarray] = None,
    ) -> Optional[Variant]:
        """Produce a compressed Variant, or None if unavailable/skipped."""
