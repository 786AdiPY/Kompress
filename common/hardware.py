"""Map a deployment target (where the compressed model will actually run) to the
compression techniques worth trying for it.

This is what makes `target_hardware` in a job manifest *drive* compression rather
than just label the benchmark: a model bound for an NVIDIA GPU should attempt the
TensorRT INT8 engine; one bound for a generic CPU box should not waste time on it
and instead lean on ONNX-Runtime INT8. The technique names here are exactly the
keys registered in ``compressors/__init__._REGISTRY`` — adding a compressor there
and referencing it here is all it takes to teach a new hardware target.
"""
from __future__ import annotations

# target_hardware id -> ordered list of compressor names to attempt.
# onnx_fp32 is always included as the portable baseline / gate reference.
HARDWARE_COMPRESSORS: dict[str, list[str]] = {
    # Generic x86/ARM CPU serving (Cloud Run, EC2, on-prem) — no GPU assumed.
    "cpu-generic":  ["onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"],
    # NVIDIA datacenter/edge GPU (A10G, T4, Jetson) — TensorRT INT8 engine.
    "nvidia-gpu":   ["onnx_fp32", "trt_int8"],
    # Sony IMX500 intelligent-vision sensor NPU — static INT8 is the closest
    # portable proxy today; swap in a dedicated MCT-backed compressor when added.
    "sony-imx500":  ["onnx_fp32", "onnx_int8_static"],
    # Generic ARM NPU / mobile accelerators — static INT8.
    "arm-npu":      ["onnx_fp32", "onnx_int8_static"],
}

# Used when a job specifies neither explicit methods nor a known hardware target.
DEFAULT_HARDWARE = "cpu-generic"


def known_targets() -> list[str]:
    return sorted(HARDWARE_COMPRESSORS)


def compressors_for(
    target_hardware: str | None,
    explicit: list[str] | None = None,
) -> list[str]:
    """Resolve the compressor list for a job.

    Precedence:
      1. an explicit ``compression.methods`` list in the job manifest (caller
         knows best — we honour it verbatim),
      2. else the mapping for ``target_hardware``,
      3. else the safe CPU default.
    """
    if explicit:
        return list(explicit)
    if target_hardware in HARDWARE_COMPRESSORS:
        return list(HARDWARE_COMPRESSORS[target_hardware])
    return list(HARDWARE_COMPRESSORS[DEFAULT_HARDWARE])
