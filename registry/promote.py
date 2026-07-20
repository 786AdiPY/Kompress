"""Declarative promotion & rollback — one operation, addressed by identity.

Deploying on consent and rolling back are the SAME action: re-point the serving
stage of a registered model to a specific, already-registered version, identified
by the MLflow run that produced it (or the git commit that run was built from).
Rollback is not a special code path — it is `promote` aimed at an older run.

    # deploy the version from a specific run (Front Door B "approve")
    python registry/promote.py --model churn --run-id <mlflow_run_id>

    # roll back: same command, older run
    python registry/promote.py --model churn --run-id <previous_run_id>

    # address by the git commit the model was built from
    python registry/promote.py --model churn --git-commit <sha>

Does not modify registry/register.py (which does first registration). Scoped to
whichever MLflow instance MLFLOW_TRACKING_URI / --tracking-uri points at.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registry import mlflow_state  # noqa: E402

GIT_COMMIT_TAG = "mlflow.source.git.commit"


def _client(tracking_uri):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=tracking_uri or os.getenv("MLFLOW_TRACKING_URI"))


def _resolve_run_id(client, model_name: str, git_commit: str) -> str:
    """Find the run for a git commit by scanning the registered model's versions."""
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if not mv.run_id:
            continue
        commit = client.get_run(mv.run_id).data.tags.get(GIT_COMMIT_TAG)
        if commit and commit.startswith(git_commit):
            return mv.run_id
    raise LookupError(f"No version of '{model_name}' from git commit {git_commit}.")


def _version_for_run(client, model_name: str, run_id: str):
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.run_id == run_id:
            return mv
    raise LookupError(f"No registered version of '{model_name}' for run {run_id}.")


def promote(model_name: str, *, run_id: str | None = None,
            git_commit: str | None = None, stage: str = "Production",
            set_status: str | None = mlflow_state.APPROVED,
            tracking_uri: str | None = None) -> dict:
    if not (run_id or git_commit):
        raise ValueError("Provide --run-id or --git-commit.")
    client = _client(tracking_uri)

    if git_commit and not run_id:
        run_id = _resolve_run_id(client, model_name, git_commit)

    mv = _version_for_run(client, model_name, run_id)
    client.transition_model_version_stage(
        name=model_name, version=mv.version, stage=stage,
        archive_existing_versions=True,
    )
    # Record consent/rollback on the run itself, so the dashboard reflects it.
    if set_status:
        mlflow_state.set_status(run_id, set_status, tracking_uri=tracking_uri)

    result = {"model": model_name, "version": mv.version, "run_id": run_id,
              "stage": stage, "status": set_status}
    print(f"Promoted '{model_name}' v{mv.version} (run {run_id[:8]}) -> {stage}"
          + (f", status={set_status}" if set_status else ""))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Registered model name.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="MLflow run id whose version to promote.")
    g.add_argument("--git-commit", help="Git commit sha the version was built from.")
    ap.add_argument("--stage", default="Production")
    ap.add_argument("--status", default=mlflow_state.APPROVED,
                    help=f"Lifecycle status to stamp on the run (default {mlflow_state.APPROVED}).")
    ap.add_argument("--tracking-uri", default=None)
    args = ap.parse_args()

    promote(args.model, run_id=args.run_id, git_commit=args.git_commit,
            stage=args.stage, set_status=args.status, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
