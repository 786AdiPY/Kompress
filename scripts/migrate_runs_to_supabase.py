"""Migrate runs from MLflow + api_runs to Supabase `run_metrics` table.

Usage:
    python scripts/migrate_runs_to_supabase.py
"""
import json
import os
import sys

def get_repo_root():
    cur = os.path.abspath(__file__)
    while os.path.dirname(cur) != cur:
        if os.path.exists(os.path.join(cur, "pyproject.toml")) or os.path.exists(os.path.join(cur, "pipeline.yaml")):
            return cur
        cur = os.path.dirname(os.path.abspath(__file__))
    return cur

REPO_ROOT = get_repo_root()
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))

import mlflow
from mlflow.tracking import MlflowClient
from supabase import create_client

def migrate():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("[ERROR] SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing in .env")
        sys.exit(1)

    client = create_client(url, key)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "self-serve-compression")
    runs_dir = os.getenv("API_RUNS_DIR", os.path.join(REPO_ROOT, "api_runs"))

    mlflow.set_tracking_uri(tracking_uri)
    m_client = MlflowClient(tracking_uri=tracking_uri)

    mlflow_runs = {}
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp:
        search_res = m_client.search_runs(experiment_ids=[exp.experiment_id])
        for r in search_res:
            mlflow_runs[r.info.run_id] = r
    print(f"[migrate] Found {len(mlflow_runs)} MLflow runs in experiment '{experiment_name}'.")

    all_run_ids = set(mlflow_runs.keys())
    if os.path.exists(runs_dir):
        all_run_ids.update([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])

    print(f"[migrate] Total unique run IDs discovered across MLflow + api_runs: {len(all_run_ids)}")

    rows_to_upsert = []
    for rid in all_run_ids:
        r_mlflow = mlflow_runs.get(rid)
        tags = r_mlflow.data.tags if r_mlflow else {}
        metrics = r_mlflow.data.metrics if r_mlflow else {}

        # Job configuration
        job = {}
        job_file = os.path.join(runs_dir, rid, "job.json")
        if os.path.exists(job_file):
            try:
                with open(job_file) as f:
                    job = json.load(f)
            except Exception:
                pass

        # Compression report
        report = {}
        art_dir = os.path.join(runs_dir, rid, "artifacts")
        if os.path.exists(art_dir):
            for sub in os.listdir(art_dir):
                rf = os.path.join(art_dir, sub, "compression_report.json")
                if os.path.exists(rf):
                    try:
                        with open(rf) as f:
                            report = json.load(f)
                        break
                    except Exception:
                        pass

        model_name = (
            tags.get("model")
            or job.get("model", {}).get("name")
            or job.get("model", {}).get("framework")
            or "unknown"
        )
        framework = tags.get("framework") or job.get("model", {}).get("framework")
        task = job.get("model", {}).get("task")
        target_hardware = tags.get("target_hardware") or job.get("target_hardware")
        status = tags.get("status") or "unknown"

        best_variant = tags.get("best_variant") or (report.get("best_variant") or {}).get("name")
        deltas = report.get("deltas", {})

        size_delta_pct = metrics.get("size_delta_pct") if metrics.get("size_delta_pct") is not None else deltas.get("size_delta_pct")
        latency_ms_delta = metrics.get("latency_ms_delta") if metrics.get("latency_ms_delta") is not None else deltas.get("latency_ms_delta")
        accuracy_delta = deltas.get("accuracy_delta")
        auc_delta = deltas.get("auc_delta")
        f1_delta = deltas.get("f1_delta")
        rmse_delta = deltas.get("rmse_delta")
        gate_passed = report.get("gate_passed") if "gate_passed" in report else (
            tags.get("gate_passed") == "True" if "gate_passed" in tags else None
        )

        row = {
            "run_id": rid,
            "model": model_name,
            "framework": framework,
            "task": task,
            "target_hardware": target_hardware,
            "status": status,
            "best_variant": best_variant,
            "size_delta_pct": size_delta_pct,
            "latency_ms_delta": latency_ms_delta,
            "accuracy_delta": accuracy_delta,
            "auc_delta": auc_delta,
            "f1_delta": f1_delta,
            "rmse_delta": rmse_delta,
            "gate_passed": gate_passed,
            "report": report if report else None,
        }
        rows_to_upsert.append(row)

    if rows_to_upsert:
        res = client.table("run_metrics").upsert(rows_to_upsert).execute()
        print(f"[migrate] Successfully migrated {len(res.data)} runs into Supabase 'run_metrics'.")
    else:
        print("[migrate] No runs found to migrate.")

if __name__ == "__main__":
    migrate()
