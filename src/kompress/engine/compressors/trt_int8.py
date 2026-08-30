"""TensorRT INT8 compressor. Requires an NVIDIA GPU + TensorRT runtime.

When TensorRT/pycuda are unavailable (laptops, CI, most dev machines) it
returns a Variant that transparently falls back to the FP32 ONNX graph, so the
rest of the pipeline still has a third variant to benchmark and serve.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import Compressor, Variant


class TrtInt8(Compressor):
    name = "trt_int8"

    def compress(self, fp32_onnx_path, out_dir, *, calib_data=None) -> Optional[Variant]:
        trt_path = os.path.join(out_dir, "model_int8.trt")
        try:
            import tensorrt  # noqa: F401
        except Exception:
            print(f"[{self.name}] TensorRT not available — falling back to FP32 ONNX.")
            return Variant(
                name=self.name, kind="onnx", path=fp32_onnx_path,
                size_kb=Variant.size_of(fp32_onnx_path),
                note="fallback: no GPU/TensorRT",
            )

        try:
            self._build_engine(fp32_onnx_path, trt_path, calib_data)
        except Exception as e:
            print(f"[{self.name}] TRT build failed ({e}); falling back to FP32 ONNX.")
            return Variant(
                name=self.name, kind="onnx", path=fp32_onnx_path,
                size_kb=Variant.size_of(fp32_onnx_path),
                note=f"fallback: {e}",
            )

        return Variant(name=self.name, kind="trt", path=trt_path,
                       size_kb=Variant.size_of(trt_path))

    def _build_engine(self, onnx_path, trt_path, calib_data):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        n_features = int(calib_data.shape[1]) if calib_data is not None else 10
        if calib_data is None:
            rng = np.random.default_rng(42)
            calib_data = rng.uniform(0, 1, (1000, n_features)).astype(np.float32)
        calib_data = np.asarray(calib_data, dtype=np.float32)

        class _Calib(trt.IInt8EntropyCalibrator2):
            def __init__(self, data, cache_file):
                super().__init__()
                import pycuda.driver as cuda
                import pycuda.autoinit  # noqa
                self._data = data
                self._idx = 0
                self._bs = 128
                self._cache = cache_file
                self._mem = cuda.mem_alloc(data[0:self._bs].nbytes)

            def get_batch_size(self):
                return self._bs

            def get_batch(self, names):
                import pycuda.driver as cuda
                if self._idx + self._bs > len(self._data):
                    return None
                batch = self._data[self._idx:self._idx + self._bs]
                cuda.memcpy_htod(self._mem, batch)
                self._idx += self._bs
                return [int(self._mem)]

            def read_calibration_cache(self):
                if os.path.exists(self._cache):
                    with open(self._cache, "rb") as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                with open(self._cache, "wb") as f:
                    f.write(cache)

        with trt.Builder(logger) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, logger) as parser:

            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = _Calib(calib_data, trt_path + ".calib")

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    raise RuntimeError("ONNX parse failed for TRT")

            profile = builder.create_optimization_profile()
            inp = network.get_input(0)
            profile.set_shape(inp.name, (1, n_features), (64, n_features), (256, n_features))
            config.add_optimization_profile(profile)

            engine_bytes = builder.build_serialized_network(network, config)
            os.makedirs(os.path.dirname(trt_path) or ".", exist_ok=True)
            with open(trt_path, "wb") as f:
                f.write(engine_bytes)

        print(f"[{self.name}] TRT INT8 engine saved -> {trt_path} "
              f"({Variant.size_of(trt_path):.1f} KB)")
