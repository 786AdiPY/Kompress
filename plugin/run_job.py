"""run_job.py — the single entrypoint any orchestrator calls to compress ONE model.

This is Front Door A. A caller (Jenkins step, Airflow BashOperator, GitHub Action,
or the Front Door B API) hands in a job manifest describing a model + its test set +
the hardware it will deploy to. run_job.py:

  1. loads + (optionally) validates the job against plugin/job.schema.json,
  2. resolves which compressors to run from target_hardware (common/hardware.py),
  3. materializes the job into a one-model pipeline.yaml (reusing the existing
     config machinery so no stage needs to change), and
  4. runs compress -> benchmark -> gate -> report on that single model,
     emitting artifacts/<name>/compression_report.json ("the Plan").

Usage:
    python plugin/run_job.py --job plugin/job.example.yaml
    docker run --rm -v $PWD:/work compression-pipeline:local \\
        python plugin/run_job.py --job /work/plugin/job.example.yaml

Exit code mirrors the gate: 0 if the winning variant passed, non-zero otherwise —
but the report is written either way so the caller always sees why.
"""
import argparse
import os
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.hardware import compressors_for, known_targets  # noqa: E402

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job.schema.json")

# schemes we don't resolve here — Front Door B's API pulls these server-side (Phase 6).
_REMOTE_SCHEMES = ("s3://", "gs://", "http://", "https://", "mlflow://")


def load_job(path: str) -> dict:
    with open(path) as f:
        return prune_none(yaml.safe_load(f))


def prune_none(job: dict) -> dict:
    """Drop null-valued keys at the job / model / test_data levels. Callers that
    serialize optional fields as null (e.g. the API's pydantic model_dump, or a
    YAML `gate:` left blank) would otherwise fail schema validation, since an
    omitted optional and an explicit null are not the same to jsonschema."""
    out = {k: v for k, v in job.items() if v is not None}
    for section in ("model", "test_data"):
        if isinstance(out.get(section), dict):
            out[section] = {k: v for k, v in out[section].items() if v is not None}
    return out


def validate_job(job: dict) -> None:
    try:
        import json
        import jsonschema
    except ImportError:
        return  # validation is best-effort; skip if jsonschema absent
    with open(SCHEMA_PATH) as f:
        jsonschema.validate(job, json.load(f))


def resolve_ref(ref: str, what: str) -> str:
    """Local paths pass through. Remote pointers are the API's job to fetch, not
    this headless CLI's — fail loudly so a caller can't think a URI 'worked'."""
    if ref.startswith(_REMOTE_SCHEMES):
        raise NotImplementedError(
            f"{what} ref '{ref}' is a remote pointer. Fetch it to a local path first, "
            f"or submit this job through the Front Door B API which resolves pointers "
            f"server-side. run_job.py operates on local paths.")
    if not os.path.isabs(ref):
        ref = os.path.join(REPO_ROOT, ref)
    if not os.path.exists(ref):
        raise FileNotFoundError(f"{what} not found: {ref}")
    return ref


def infer_features(test_csv: str, target: str) -> list:
    """Convenience: if the job omits an explicit feature schema, take every column
    of the test set except the target as a float feature. Keeps the caller's job
    manifest minimal — 'just the test dataset', as intended."""
    import pandas as pd
    cols = list(pd.read_csv(test_csv, nrows=1).columns)
    return [{"name": c, "dtype": "float"} for c in cols if c != target]


def build_pipeline_config(job: dict, model_path: str, test_path: str) -> dict:
    model = job["model"]
    name = model.get("name") or model.get("framework") or "job_model"
    target = model.get("target", "target")
    features = model.get("features") or infer_features(test_path, target)

    methods = compressors_for(
        job.get("target_hardware"),
        explicit=(job.get("compression") or {}).get("methods"),
    )

    return {
        "project": job.get("project", "compression-job"),
        "gate": job.get("gate", {}),
        "models": [{
            "name": name,
            "framework": model["framework"],
            "task": model["task"],
            "artifact": model_path,
            "num_classes": int(model.get("num_classes", 2)),
            "target": target,
            "trainable": False,
            "data": {"test": test_path},
            "features": features,
            "compression": {"methods": methods},
            "gate": job.get("gate", {}),
        }],
    }, name, methods


def run_stage(script: str, env: dict) -> int:
    print(f"\n──▶ {script}")
    return subprocess.run([sys.executable, os.path.join(REPO_ROOT, script)],
                          cwd=REPO_ROOT, env=env).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="Path to a job manifest (yaml/json).")
    ap.add_argument("--artifacts-dir", default=os.path.join(REPO_ROOT, "artifacts"))
    ap.add_argument("--config-out", default=None,
                    help="Where to write the materialized pipeline.yaml (default: scratch in artifacts).")
    args = ap.parse_args()

    job = load_job(args.job)
    validate_job(job)

    model_path = resolve_ref(job["model"]["ref"], "model")
    test_path = resolve_ref(job["test_data"]["ref"], "test_data")
    hardware = job.get("target_hardware", "cpu-generic")
    if hardware not in known_targets():
        print(f"[warn] target_hardware '{hardware}' unknown; compressor selection falls "
              f"back to CPU default. Known: {known_targets()}")

    config, name, methods = build_pipeline_config(job, model_path, test_path)
    print(f"Job: model='{name}' framework={job['model']['framework']} "
          f"hardware={hardware} compressors={methods}")

    os.makedirs(args.artifacts_dir, exist_ok=True)
    config_out = args.config_out or os.path.join(args.artifacts_dir, f"_job_{name}.pipeline.yaml")
    with open(config_out, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    env = {**os.environ,
           "PIPELINE_CONFIG": config_out,
           "ARTIFACTS_DIR": args.artifacts_dir,
           # Scope the models index into THIS job's artifacts dir; otherwise every
           # job writes the repo-default artifacts/models.json and concurrent jobs
           # (e.g. from the API) clobber each other.
           "MODELS_INDEX": os.path.join(args.artifacts_dir, "models.json"),
           "TARGET_HARDWARE": hardware,
           "PYTHONPATH": REPO_ROOT}

    for script in ("compress/compress.py", "benchmark/benchmark.py"):
        if run_stage(script, env) != 0:
            print(f"\nStage {script} failed — aborting job.")
            sys.exit(2)

    gate_rc = run_stage("gate/gate.py", env)          # non-zero if gate fails
    run_stage("report/generate_report.py", env)        # always emit the Plan

    report = os.path.join(args.artifacts_dir, name, "compression_report.json")
    print(f"\n{'GATE PASSED' if gate_rc == 0 else 'GATE FAILED'} — Plan: {report}")
    sys.exit(gate_rc)


if __name__ == "__main__":
    main()
