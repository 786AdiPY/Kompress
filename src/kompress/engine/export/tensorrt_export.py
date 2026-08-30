"""TensorRT export for NVIDIA GPU / Jetson. Builds a serialized TRT engine from
ONNX when a TensorRT runtime is present; otherwise stays 'unavailable' and points
the user at converting on their own GPU box with trtexec."""
from __future__ import annotations

import os

from .base import ExportTarget

_TRTEXEC_HINT = ("No TensorRT runtime on this host. Download the ONNX and build the "
                 "engine on your NVIDIA machine: "
                 "`trtexec --onnx=model.onnx --saveEngine=model.trt --int8`.")


class TensorRTExport(ExportTarget):
    format = "tensorrt"
    label = "TensorRT engine (NVIDIA GPU / Jetson)"
    devices = ["NVIDIA GPU", "Jetson"]
    extension = ".trt"

    def __init__(self):
        try:
            import tensorrt  # noqa: F401
            self.available = True
        except Exception:
            self.available = False
            self.unavailable_reason = _TRTEXEC_HINT

    def export(self, onnx_path: str, out_dir: str) -> str:
        if not self.available:
            raise RuntimeError(_TRTEXEC_HINT)
        import tensorrt as trt

        os.makedirs(out_dir, exist_ok=True)
        engine_path = os.path.join(out_dir, "model.trt")
        logger = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(logger) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, logger) as parser:
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    raise RuntimeError("ONNX parse failed for TensorRT export")
            config = builder.create_builder_config()
            engine = builder.build_serialized_network(network, config)
            with open(engine_path, "wb") as f:
                f.write(engine)
        return engine_path

    def runtime_hint(self) -> str:
        return "Load with tensorrt.Runtime(...).deserialize_cuda_engine(open('model.trt','rb').read())"
