"""The canonical "run one compression job" logic, shared by the worker and by the
API's inline fallback. FastAPI-free on purpose so a worker process can import it
without pulling in the web layer.

execute_job() is the whole runtime pipeline for an uploaded model: resolve the
pointers, run the compression engine (plugin/run_job.py) in a run-scoped artifacts
dir, then STORE THE RESULT IN MLFLOW — log the deltas, register the winning ONNX to
the Model Registry (so /approve can promote it), and advance the run's status tag.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from registry import mlflow_state  # noqa: E402

# Same env contract the API uses — one deployment configures both processes.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "self-serve-compression")
PROJECT = os.getenv("PROJECT", "compression")
RUNS_DIR = os.getenv("API_RUNS_DIR", os.path.join(REPO_ROOT, "api_runs"))
ALLOW_LOCAL_PATHS = os.getenv("API_ALLOW_LOCAL_PATHS", "0") == "1"
_REMOTE_SCHEMES = ("s3://", "gs://", "http://", "https://", "mlflow://")


class PointerError(Exception):
    """Raised for a bad/disallowed data pointer. Carries an HTTP-ish status_code so
    the API can translate it to a response; the worker just fails the job."""
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def registered_name(model_name: str) -> str:
    return f"{PROJECT}-{model_name}"


# ── pointer resolution (the security boundary) ───────────────────────────────
def check_ref_policy(ref: str, what: str) -> None:
    """Policy check (no IO): only remote pointers are accepted unless local paths
    are explicitly enabled. Raw uploads never reach here — the API takes refs only."""
    if ref.startswith("s3://"):
        return
    if ref.startswith(_REMOTE_SCHEMES):
        raise PointerError(501, f"Fetching {what} scheme not implemented: {ref}")
    if not ALLOW_LOCAL_PATHS:
        raise PointerError(
            400, f"{what} must be a remote pointer (s3://, mlflow://). Local paths "
                 f"are disabled; set API_ALLOW_LOCAL_PATHS=1 for single-tenant self-host.")


def resolve_pointer(ref: str, dest_dir: str, what: str) -> str:
    """Turn a submitted pointer into a local file. Re-checks policy, then fetches."""
    check_ref_policy(ref, what)
    os.makedirs(dest_dir, exist_ok=True)
    if ref.startswith("s3://"):
        return _fetch_s3(ref, dest_dir, what)
    path = ref if os.path.isabs(ref) else os.path.join(REPO_ROOT, ref)
    if not os.path.exists(path):
        raise PointerError(404, f"{what} not found: {path}")
    return path


def _fetch_s3(ref: str, dest_dir: str, what: str) -> str:
    try:
        import boto3
    except ImportError:
        raise PointerError(501, "boto3 not installed; cannot fetch s3:// refs.")
    bucket, _, key = ref[len("s3://"):].partition("/")
    local = os.path.join(dest_dir, os.path.basename(key))
    boto3.client("s3").download_file(bucket, key, local)
    return local


# ── storing the result in MLflow ─────────────────────────────────────────────
def log_report(run_id: str, report: dict):
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=TRACKING_URI)
    d = report["deltas"]
    for k in ("size_delta_pct", "latency_ms_delta", "speedup_vs_native"):
        if d.get(k) is not None:
            client.log_metric(run_id, k, float(d[k]))
    mlflow_state.tag_run(run_id, extra={
        "best_variant": report["best_variant"]["name"],
        "gate_passed": report["gate_passed"],
    }, tracking_uri=TRACKING_URI)


def register_best(run_id: str, model_name: str, art_dir: str, report: dict):
    """Register the winning ONNX to the MLflow Model Registry, linked to this run, so
    /approve can promote that exact version. ONNX is the universal registry artifact
    (mirrors registry/register.py). Best-effort — failure is tagged, not fatal."""
    try:
        import mlflow
        import mlflow.onnx
        import onnx
        manifest = json.load(open(os.path.join(art_dir, model_name, "variants.json")))
        best = report["best_variant"]["name"]
        variants = manifest.get("variants", [])
        onnx_path = next((v["path"] for v in variants
                          if v.get("name") == best and str(v.get("path", "")).endswith(".onnx")), None)
        if onnx_path is None:
            onnx_path = next((v["path"] for v in variants
                              if str(v.get("path", "")).endswith(".onnx")), None)
        if onnx_path is None:
            return
        onnx_path = onnx_path if os.path.isabs(onnx_path) else os.path.join(REPO_ROOT, onnx_path)
        mlflow.set_tracking_uri(TRACKING_URI)
        onnx_model = onnx.load(onnx_path)
        reg_name = registered_name(model_name)
        with mlflow.start_run(run_id=run_id):
            try:
                mlflow.onnx.log_model(onnx_model, name="model", registered_model_name=reg_name)
            except TypeError:
                mlflow.onnx.log_model(onnx_model, artifact_path="model", registered_model_name=reg_name)
    except Exception as e:  # noqa: BLE001
        mlflow_state.tag_run(run_id, extra={"register_error": str(e)[:200]}, tracking_uri=TRACKING_URI)


# ── the job itself ───────────────────────────────────────────────────────────
def execute_job(run_id: str, job: dict, model_name: str) -> str:
    """Run one compression job to completion and reflect the outcome in MLflow.
    Returns the terminal status. Safe to call from a worker process or an inline
    background task — it owns the whole lifecycle and never raises."""
    run_dir = os.path.join(RUNS_DIR, run_id)
    art_dir = os.path.join(run_dir, "artifacts")
    try:
        model_path = resolve_pointer(job["model"]["ref"], run_dir, "model")
        test_path = resolve_pointer(job["test_data"]["ref"], run_dir, "test_data")
        job = {**job, "model": {**job["model"], "ref": model_path},
               "test_data": {"ref": test_path}}
        job_file = os.path.join(run_dir, "job.json")
        with open(job_file, "w") as f:
            json.dump(job, f)

        env = {**os.environ, "MLFLOW_TRACKING_URI": TRACKING_URI,
               "MLFLOW_RUN_ID": run_id, "TARGET_HARDWARE": job["target_hardware"]}
        rc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "plugin", "run_job.py"),
             "--job", job_file, "--artifacts-dir", art_dir],
            cwd=REPO_ROOT, env=env,
        ).returncode

        report_path = os.path.join(art_dir, model_name, "compression_report.json")
        if not os.path.exists(report_path):
            mlflow_state.set_status(run_id, mlflow_state.FAILED, tracking_uri=TRACKING_URI)
            return mlflow_state.FAILED

        report = json.load(open(report_path))
        log_report(run_id, report)
        if rc == 0:
            register_best(run_id, model_name, art_dir, report)
        status = mlflow_state.PENDING_APPROVAL if rc == 0 else mlflow_state.REJECTED
        mlflow_state.set_status(run_id, status, tracking_uri=TRACKING_URI)
        return status
    except Exception as e:  # noqa: BLE001 — any failure becomes a terminal status
        mlflow_state.tag_run(run_id, status=mlflow_state.FAILED,
                             extra={"error": str(e)[:250]}, tracking_uri=TRACKING_URI)
        return mlflow_state.FAILED
