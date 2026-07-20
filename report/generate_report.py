"""Generate the compression_report.json ("the Plan") for each compressed model.

Pure aggregation: reads the artifacts every existing stage already wrote —
artifacts/<name>/variants.json (compress), benchmark_results.json (benchmark),
gate_report.json (gate) — picks the best gate-passing compressed variant, and
records its size / latency / accuracy deltas versus the native baseline plus a
sha256 of the base model for provenance. Writes artifacts/<name>/compression_report.json.

Front Door B shows this to a human before they consent to deploy; Front Door A
can archive it with no MLflow instance at all. Touches no existing stage.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config  # noqa: E402

# Where the model runs, set by plugin/run_job.py per job; the batch pipeline
# leaves it at the portable default. Compression metrics are only meaningful
# alongside this — see report/report.schema.json.
TARGET_HARDWARE = os.getenv("TARGET_HARDWARE", "cpu-generic")

_FORMAT = {
    "onnx_fp32": "onnx-fp32",
    "onnx_int8_dynamic": "onnx-int8-dynamic",
    "onnx_int8_static": "onnx-int8-static",
    "trt_int8": "tensorrt-int8",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _sha256(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_native(results: list) -> dict | None:
    for r in results:
        if r.get("kind") == "native" or r["model"].startswith("native_"):
            return r
    return results[0] if results else None


def _passing_variants(gate: dict) -> set:
    return {c["variant"] for c in gate.get("checks", []) if c.get("passed")}


def _pick_best(results: list, native: dict, passing: set) -> dict | None:
    """Best = smallest gate-passing COMPRESSED variant, tie-broken by lowest
    latency. Compression's headline is size; latency is surfaced as a delta so a
    reviewer sees the trade honestly. Returns None if nothing compressed passed."""
    compressed = [
        r for r in results
        if r is not native and r.get("kind") != "native"
        and not r["model"].startswith("native_") and r["model"] in passing
    ]
    if not compressed:
        return None
    return min(compressed, key=lambda r: (r.get("model_size_kb", 0.0), r["latency_ms"]))


def _delta(best: dict, native: dict, key: str):
    if key in best and key in native:
        return round(best[key] - native[key], 4)
    return None


def build_report(spec, target_hardware: str = TARGET_HARDWARE) -> dict | None:
    manifest = _load(spec.manifest_path)
    results = _load(os.path.join(spec.out_dir, "benchmark_results.json"))
    gate = _load(os.path.join(spec.out_dir, "gate_report.json"))

    native = _find_native(results)
    if native is None:
        print(f"[{spec.name}] no native baseline in benchmark results — skipping report.")
        return None

    passing = _passing_variants(gate)
    best = _pick_best(results, native, passing)
    gate_passed = best is not None

    # If nothing compressed passed the gate, still emit a report (gate_passed=False)
    # pointing at the smallest compressed variant so the reviewer sees why.
    if best is None:
        compressed = [r for r in results if r.get("kind") != "native"
                      and not r["model"].startswith("native_")]
        best = min(compressed, key=lambda r: r.get("model_size_kb", 0.0)) if compressed else native

    native_size = native.get("model_size_kb", 0.0) or 0.0
    best_size = best.get("model_size_kb", 0.0) or 0.0
    size_delta_pct = round((best_size - native_size) / native_size * 100, 2) if native_size else 0.0

    report = {
        "model": spec.name,
        "framework": spec.framework,
        "task": spec.task,
        "target_hardware": target_hardware,
        "base_model": {
            "path": manifest["native"]["path"],
            "hash": _sha256(manifest["native"]["path"]),
            "size_kb": native_size,
            "mlflow_run_id": os.getenv("MLFLOW_RUN_ID"),  # null unless a run set it
        },
        "best_variant": {
            "name": best["model"],
            "kind": best.get("kind", ""),
            "format": _FORMAT.get(best["model"], best["model"].replace("_", "-")),
            "size_kb": best_size,
            # Preserve any compressor note (e.g. "fallback: no GPU/TensorRT") so the
            # reported format isn't read as a guarantee the target engine was used.
            "note": best.get("note", ""),
        },
        "deltas": {
            "size_delta_pct": size_delta_pct,
            "latency_ms_delta": round(best["latency_ms"] - native["latency_ms"], 4),
            "speedup_vs_native": best.get("speedup_vs_native", 1.0),
            "accuracy_delta": _delta(best, native, "accuracy"),
            "auc_delta": _delta(best, native, "auc"),
            "f1_delta": _delta(best, native, "f1"),
            "rmse_delta": _delta(best, native, "rmse"),
        },
        "variants": results,
        "gate_passed": gate_passed,
        "gate_report": gate,
    }
    return report


def write_report(spec, report: dict) -> str:
    out = os.path.join(spec.out_dir, "compression_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    d = report["deltas"]
    print(f"[{spec.name}] Plan -> {out}  "
          f"best={report['best_variant']['name']} "
          f"size={d['size_delta_pct']:+.1f}% latency={d['latency_ms_delta']:+.2f}ms "
          f"gate={'PASS' if report['gate_passed'] else 'FAIL'}")
    return out


def main():
    cfg = load_config()
    wrote = 0
    for spec in cfg.models:
        needed = ["variants.json", "benchmark_results.json", "gate_report.json"]
        if not all(os.path.exists(os.path.join(spec.out_dir, n)) for n in needed):
            print(f"[{spec.name}] missing benchmark/gate artifacts — skipping report.")
            continue
        report = build_report(spec)
        if report is not None:
            write_report(spec, report)
            wrote += 1
    print(f"\nGenerated {wrote} compression report(s).")


if __name__ == "__main__":
    main()
