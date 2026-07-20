"""Front Door B — the self-serve compression API the dashboard is a client of.

Every UI screen maps to one endpoint here; the UI holds no compression or
promotion logic of its own. State lives entirely in MLflow tags (see
registry/mlflow_state.py) — there is no separate approvals database.

  POST /runs               submit a job (pointer refs ONLY, never raw uploads)
  GET  /runs               review queue, filterable by ?status=
  GET  /runs/{id}          one run's status + summary
  GET  /runs/{id}/report   the compression_report.json ("the Plan")
  GET  /runs/{id}/artifact download the winning compressed variant
  GET  /runs/{id}/export   export to a device format (onnx|tensorrt|tflite|coreml)
  POST /runs/{id}/approve  consent -> promote to Production (registry/promote.py)
  POST /runs/{id}/reject   reject
  POST /runs/{id}/rollback re-point Production to a previous run
  GET  /models             registered models w/ Production & latest versions
  GET  /models/{name}/versions  a model's versions + lifecycle status, newest first
  GET  /export-formats     device export targets for the download UI
  GET  /hardware-targets   populate the hardware dropdown (common/hardware.py)
  GET  /health

Compression execution is delegated to worker/executor.py so it can run either
in-process (KOMPRESS_EXECUTION=inline, the default) or on a separate worker pool
(KOMPRESS_EXECUTION=queue) — the API stays a thin enqueuer either way.

This runs against the PLATFORM's OWN MLflow (MLFLOW_TRACKING_URI), which is the
system of record for self-serve runs — distinct from any MLflow a Front Door A
caller might use.
"""
from __future__ import annotations

import json
import os
import sys

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.hardware import known_targets                 # noqa: E402
from registry import mlflow_state, promote                 # noqa: E402
from export import get_exporter, list_exporters            # noqa: E402
from worker.executor import (                              # noqa: E402
    TRACKING_URI, EXPERIMENT, RUNS_DIR, PointerError,
    check_ref_policy, execute_job, registered_name,
)
from worker.queue import get_queue                          # noqa: E402

# How submitted jobs run:
#   inline (default) — a FastAPI BackgroundTask runs the compression in-process.
#   queue            — enqueue for a separate worker pool (worker/worker.py):
#                      scalable, isolated, survives an API restart.
EXECUTION_MODE = os.getenv("KOMPRESS_EXECUTION", "inline")

app = FastAPI(title="Kompress — Self-Serve API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ── request models ───────────────────────────────────────────────────────────
class Ref(BaseModel):
    ref: str = Field(..., description="Pointer to data (s3://, mlflow://, ...). Not raw bytes.")


class ModelSpecIn(BaseModel):
    name: str | None = None
    ref: str
    framework: str
    task: str
    num_classes: int = 2
    target: str = "target"
    features: list[dict] | None = None


class JobIn(BaseModel):
    model: ModelSpecIn
    test_data: Ref
    target_hardware: str
    compression: dict | None = None
    gate: dict | None = None


# ── run creation + artifact lookup ───────────────────────────────────────────
def _new_run(job: dict, model_name: str) -> str:
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"job-{model_name}") as run:
        run_id = run.info.run_id
    mlflow_state.tag_run(run_id, source=mlflow_state.SOURCE_UI,
                         status=mlflow_state.PENDING_GATE,
                         extra={"model": model_name,
                                "framework": job["model"]["framework"],
                                "target_hardware": job["target_hardware"]},
                         tracking_uri=TRACKING_URI)
    return run_id


def _run_dir_report(run_id: str, model_name: str | None = None) -> str:
    base = os.path.join(RUNS_DIR, run_id, "artifacts")
    if model_name:
        return os.path.join(base, model_name, "compression_report.json")
    for name in os.listdir(base) if os.path.isdir(base) else []:
        cand = os.path.join(base, name, "compression_report.json")
        if os.path.exists(cand):
            return cand
    raise HTTPException(404, "No report for this run yet.")


def _run_onnx_path(run_id: str) -> str:
    """Resolve a completed run's best-variant ONNX to an absolute path — the input
    every export target converts from. Shared by /export."""
    report_path = _run_dir_report(run_id)                 # 404s if no report yet
    model_dir = os.path.dirname(report_path)
    report = json.load(open(report_path))
    variants = json.load(open(os.path.join(model_dir, "variants.json"))).get("variants", [])
    best = report["best_variant"]["name"]
    onnx = next((v["path"] for v in variants
                 if v.get("name") == best and str(v.get("path", "")).endswith(".onnx")), None)
    if onnx is None:
        onnx = next((v["path"] for v in variants if str(v.get("path", "")).endswith(".onnx")), None)
    if onnx is None:
        raise HTTPException(409, "No ONNX variant available to export for this run.")
    return onnx if os.path.isabs(onnx) else os.path.join(REPO_ROOT, onnx)


def _get_run(run_id: str):
    from mlflow.tracking import MlflowClient
    try:
        return MlflowClient(tracking_uri=TRACKING_URI).get_run(run_id)
    except Exception:
        raise HTTPException(404, "Unknown run.")


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "execution": EXECUTION_MODE}


@app.get("/hardware-targets")
def hardware_targets():
    return {"targets": known_targets()}


@app.post("/runs", status_code=202)
def submit_run(job: JobIn, background: BackgroundTasks):
    payload = job.model_dump()
    # Validate ref policy synchronously so a bad pointer 400/501s now, not later.
    try:
        check_ref_policy(payload["model"]["ref"], "model")
        check_ref_policy(payload["test_data"]["ref"], "test_data")
    except PointerError as e:
        raise HTTPException(e.status_code, str(e))

    model_name = payload["model"].get("name") or payload["model"]["framework"]
    run_id = _new_run(payload, model_name)

    if EXECUTION_MODE == "queue":
        get_queue().enqueue({"run_id": run_id, "job": payload, "model_name": model_name})
    else:  # inline: run in-process
        background.add_task(execute_job, run_id, payload, model_name)

    return {"run_id": run_id, "status": mlflow_state.PENDING_GATE,
            "model": model_name, "execution": EXECUTION_MODE}


@app.get("/runs")
def list_runs(status: str | None = None):
    return {"runs": mlflow_state.list_by_tag(
        status=status, source=mlflow_state.SOURCE_UI,
        experiment=EXPERIMENT, tracking_uri=TRACKING_URI)}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    status = mlflow_state.get_status(run_id, tracking_uri=TRACKING_URI)
    if status is None:
        raise HTTPException(404, "Unknown run.")
    return {"run_id": run_id, "status": status}


@app.get("/runs/{run_id}/report")
def get_report(run_id: str):
    return json.load(open(_run_dir_report(run_id)))


@app.get("/runs/{run_id}/artifact")
def get_artifact(run_id: str):
    report_path = _run_dir_report(run_id)
    model_dir = os.path.dirname(report_path)
    report = json.load(open(report_path))
    variants = json.load(open(os.path.join(model_dir, "variants.json")))
    best = report["best_variant"]["name"]
    for v in variants["variants"]:
        if v["name"] == best:
            path = v["path"] if os.path.isabs(v["path"]) else os.path.join(REPO_ROOT, v["path"])
            return FileResponse(path, filename=os.path.basename(path))
    raise HTTPException(404, "Best variant artifact not found.")


@app.get("/export-formats")
def export_formats(hardware: str | None = None):
    """Device export targets for the 'Download for device' UI; flags the format
    recommended for the given hardware."""
    return {"formats": list_exporters(hardware)}


@app.get("/runs/{run_id}/export")
def export_model(run_id: str, format: str = "onnx"):
    """Convert the run's compressed ONNX into a device-specific deployable format
    and return it as a download (local / IoT / mobile self-deploy)."""
    try:
        exporter = get_exporter(format)
    except KeyError as e:
        raise HTTPException(400, str(e))
    if not exporter.available:
        raise HTTPException(501, exporter.unavailable_reason or f"{format} export unavailable here.")
    onnx_path = _run_onnx_path(run_id)
    out_dir = os.path.join(RUNS_DIR, run_id, "export", format)
    try:
        artifact = exporter.export(onnx_path, out_dir)
    except (RuntimeError, NotImplementedError) as e:
        raise HTTPException(501, str(e))
    return FileResponse(artifact, filename=os.path.basename(artifact))


@app.post("/runs/{run_id}/approve")
def approve(run_id: str):
    model = mlflow_state._summarize(_get_run(run_id)).get("model") or ""
    try:
        result = promote.promote(registered_name(model), run_id=run_id,
                                 set_status=mlflow_state.APPROVED, tracking_uri=TRACKING_URI)
    except LookupError as e:
        raise HTTPException(409, str(e))
    return {"approved": True, **result}


@app.post("/runs/{run_id}/reject")
def reject(run_id: str):
    mlflow_state.set_status(run_id, mlflow_state.REJECTED, tracking_uri=TRACKING_URI)
    return {"rejected": True, "run_id": run_id}


@app.post("/runs/{run_id}/rollback")
def rollback(run_id: str, to_run_id: str):
    """Roll a model's Production stage back to a previous run — promote, older run."""
    model = mlflow_state._summarize(_get_run(to_run_id)).get("model") or ""
    try:
        result = promote.promote(registered_name(model), run_id=to_run_id,
                                 set_status=mlflow_state.APPROVED, tracking_uri=TRACKING_URI)
    except LookupError as e:
        raise HTTPException(409, str(e))
    return {"rolled_back_to": to_run_id, **result}


@app.get("/models")
def list_models():
    """The model catalog: each registered model with its Production and latest versions."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=TRACKING_URI)
    models = []
    for rm in client.search_registered_models():
        versions = client.search_model_versions(f"name='{rm.name}'")
        prod = next((v for v in versions if v.current_stage == "Production"), None)
        latest = max(versions, key=lambda v: int(v.version)) if versions else None
        models.append({
            "name": rm.name,
            "production_version": prod.version if prod else None,
            "production_run_id": prod.run_id if prod else None,
            "latest_version": latest.version if latest else None,
        })
    return {"models": models}


@app.get("/models/{name}/versions")
def list_model_versions(name: str):
    """All registered versions of one model, newest first, each with its lifecycle status."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{name}'")
    out = []
    for v in sorted(versions, key=lambda v: int(v.version), reverse=True):
        status = None
        if v.run_id:
            try:
                status = mlflow_state.get_status(v.run_id, tracking_uri=TRACKING_URI)
            except Exception:  # noqa: BLE001 — a missing/deleted run must not break the list
                status = None
        out.append({
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "status": status,
        })
    return {"versions": out}
