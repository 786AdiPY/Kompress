"""MLflow-tags-as-state: the approval lifecycle without a second database.

Every run a front door produces carries two tags on its MLflow run:
  source : where it came from  — 'orchestrator-plugin' (A) | 'ui-upload' (B)
  status : where it is in the flow — pending_gate -> pending_approval
                                     -> approved | rejected  (or 'failed')

Front Door B's dashboard *is* `list_by_tag(status='pending_approval')`; the
approve/reject/rollback buttons are `set_status(...)`. No approvals table exists.

The tracking URI is always passed in or read from MLFLOW_TRACKING_URI — never a
hardcoded global — because each front door talks to its OWN MLflow instance
(A: the caller's, if any; B: the platform's dedicated one).
"""
from __future__ import annotations

import os
from typing import Optional

SOURCE_TAG = "source"
STATUS_TAG = "status"

# lifecycle states (a small, closed set)
PENDING_GATE = "pending_gate"
PENDING_APPROVAL = "pending_approval"
APPROVED = "approved"
REJECTED = "rejected"
FAILED = "failed"
STATES = {PENDING_GATE, PENDING_APPROVAL, APPROVED, REJECTED, FAILED}

# sources
SOURCE_PLUGIN = "orchestrator-plugin"
SOURCE_UI = "ui-upload"


def _client(tracking_uri: Optional[str] = None):
    from mlflow.tracking import MlflowClient
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    return MlflowClient(tracking_uri=uri)


def tag_run(run_id: str, *, source: Optional[str] = None,
            status: Optional[str] = None, extra: Optional[dict] = None,
            tracking_uri: Optional[str] = None) -> None:
    """Set source/status (and any extra tags) on an existing run."""
    if status is not None and status not in STATES:
        raise ValueError(f"Unknown status '{status}'. Valid: {sorted(STATES)}")
    client = _client(tracking_uri)
    if source is not None:
        client.set_tag(run_id, SOURCE_TAG, source)
    if status is not None:
        client.set_tag(run_id, STATUS_TAG, status)
    for k, v in (extra or {}).items():
        client.set_tag(run_id, k, str(v))


def set_status(run_id: str, status: str, tracking_uri: Optional[str] = None) -> None:
    tag_run(run_id, status=status, tracking_uri=tracking_uri)


def get_status(run_id: str, tracking_uri: Optional[str] = None) -> Optional[str]:
    run = _client(tracking_uri).get_run(run_id)
    return run.data.tags.get(STATUS_TAG)


def _experiment_ids(client, experiment: Optional[str]):
    if experiment:
        exp = client.get_experiment_by_name(experiment)
        return [exp.experiment_id] if exp else []
    return [e.experiment_id for e in client.search_experiments()]


def list_by_tag(*, status: Optional[str] = None, source: Optional[str] = None,
                experiment: Optional[str] = None, max_results: int = 200,
                tracking_uri: Optional[str] = None) -> list[dict]:
    """Return runs matching the given status/source tags, newest first — the
    query the dashboard's review queue is built on."""
    client = _client(tracking_uri)
    clauses = []
    if status:
        clauses.append(f"tags.{STATUS_TAG} = '{status}'")
    if source:
        clauses.append(f"tags.{SOURCE_TAG} = '{source}'")
    filter_string = " and ".join(clauses)

    exp_ids = _experiment_ids(client, experiment)
    if not exp_ids:
        return []
    runs = client.search_runs(
        experiment_ids=exp_ids, filter_string=filter_string,
        order_by=["attribute.start_time DESC"], max_results=max_results,
    )
    return [_summarize(r) for r in runs]


def _summarize(run) -> dict:
    t, m = run.data.tags, run.data.metrics
    return {
        "run_id": run.info.run_id,
        "status": t.get(STATUS_TAG),
        "source": t.get(SOURCE_TAG),
        "model": t.get("model"),
        "framework": t.get("framework"),
        "target_hardware": t.get("target_hardware"),
        "best_variant": t.get("best_variant"),
        "size_delta_pct": m.get("size_delta_pct"),
        "latency_ms_delta": m.get("latency_ms_delta"),
        "start_time": run.info.start_time,
    }
