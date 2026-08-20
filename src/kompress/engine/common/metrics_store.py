"""Optional metrics sink: mirrors each run's compression report into a Supabase
Postgres table (`run_metrics`) so results are queryable outside MLflow.

Configured via SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY; if either is unset this
is a no-op (MLflow stays the source of truth either way). Best-effort like
worker/executor.py's register_best() — a write failure is logged, never fatal,
since Supabase being unreachable shouldn't fail the compression job itself.
"""
from __future__ import annotations

import os

_client = None
_client_checked = False


def _get_client():
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        return None
    from supabase import create_client
    _client = create_client(url, key)
    return _client


def write_run_metrics(run_id: str, model_name: str, job: dict, report: dict, status: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        d = report.get("deltas", {})
        row = {
            "run_id": run_id,
            "model": model_name,
            "framework": job.get("model", {}).get("framework"),
            "task": job.get("model", {}).get("task"),
            "target_hardware": job.get("target_hardware"),
            "status": status,
            "best_variant": (report.get("best_variant") or {}).get("name"),
            "size_delta_pct": d.get("size_delta_pct"),
            "latency_ms_delta": d.get("latency_ms_delta"),
            "accuracy_delta": d.get("accuracy_delta"),
            "auc_delta": d.get("auc_delta"),
            "f1_delta": d.get("f1_delta"),
            "rmse_delta": d.get("rmse_delta"),
            "gate_passed": report.get("gate_passed"),
            "report": report,
        }
        client.table("run_metrics").upsert(row).execute()
    except Exception as e:  # noqa: BLE001 — never fail the job over a metrics mirror
        print(f"[metrics_store] WARNING: failed to write run {run_id} to Supabase: {e}", flush=True)
