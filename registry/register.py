"""Register the best gate-passing variant of EACH model to the MLflow Model
Registry. ONNX is the universal, framework-agnostic registry artifact.
"""
import json
import os
import sys

import mlflow
import mlflow.onnx
import onnx
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config  # noqa: E402

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def pick_best(results, gate):
    passed = {c["variant"] for c in gate.get("checks", []) if c["passed"]}
    passed.add(gate.get("baseline", ""))
    candidates = [r for r in results if r["model"] in passed] or results
    return min(candidates, key=lambda r: r["latency_ms"])


def resolve_onnx_path(best, manifest):
    for v in manifest["variants"]:
        if v["name"] == best["model"] and v["path"].endswith(".onnx"):
            return v["path"]
    for v in manifest["variants"]:
        if v["name"] == "onnx_fp32" and v["path"].endswith(".onnx"):
            return v["path"]
    for v in manifest["variants"]:
        if v["path"].endswith(".onnx"):
            return v["path"]
    raise FileNotFoundError("No ONNX variant available to register.")


def register_one(spec, cfg, client):
    manifest = load_json(spec.manifest_path)
    results = load_json(os.path.join(spec.out_dir, "benchmark_results.json"))
    gate = load_json(os.path.join(spec.out_dir, "gate_report.json"))

    if not gate.get("passed"):
        print(f"[{spec.name}] gate failed — not registering.")
        return

    best = pick_best(results, gate)
    registry_name = f"{cfg.project}-{spec.name}"
    onnx_model = onnx.load(resolve_onnx_path(best, manifest))
    experiment = os.getenv("MLFLOW_EXPERIMENT", f"{cfg.project}-compression")
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"register-{spec.name}"):
        mlflow.set_tag("model", spec.name)
        mlflow.set_tag("framework", spec.framework)
        mlflow.set_tag("best_variant", best["model"])
        mlflow.log_metric("best_latency_ms", best["latency_ms"])
        mlflow.log_metric("best_speedup", best.get("speedup_vs_native", 1.0))
        for m in ("accuracy", "auc", "f1", "rmse", "mae"):
            if m in best:
                mlflow.log_metric(f"best_{m}", best[m])
        try:
            mlflow.onnx.log_model(onnx_model, name="model", registered_model_name=registry_name)
        except TypeError:
            mlflow.onnx.log_model(onnx_model, artifact_path="model", registered_model_name=registry_name)

    versions = client.search_model_versions(f"name='{registry_name}'")
    mv = max(versions, key=lambda v: int(v.version))
    client.update_model_version(
        name=registry_name, version=mv.version,
        description=(f"Best variant: {best['model']} | framework={spec.framework} "
                     f"| latency={best['latency_ms']:.2f}ms "
                     f"| speedup={best.get('speedup_vs_native', 1.0):.2f}x vs native"),
    )
    try:
        client.transition_model_version_stage(
            name=registry_name, version=mv.version,
            stage="Production", archive_existing_versions=True)
    except Exception as e:
        print(f"    (stage transition skipped: {e})")
    print(f"[{spec.name}] registered '{registry_name}' v{mv.version} -> Production "
          f"(best={best['model']})")


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    for spec in cfg.models:
        if not os.path.exists(os.path.join(spec.out_dir, "gate_report.json")):
            print(f"[{spec.name}] no gate report — skipping.")
            continue
        try:
            register_one(spec, cfg, client)
        except Exception as e:
            print(f"[{spec.name}] registration failed: {e}")


if __name__ == "__main__":
    main()
