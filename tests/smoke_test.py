"""Kompress platform smoke test.

This is PLATFORM CI: it verifies Kompress's OWN code — package imports, the JSON
schemas, and that the compression ENGINE runs end to end on a tiny throwaway
fixture producing a schema-valid compression_report.json.

It deliberately does NOT compress the demo/customer models Kompress orchestrates
(churn, house_price). Compressing a specific model is runtime work triggered via
the API / plugin — not the platform's build. So the fixture here is a 4-feature
toy model generated on the fly, thrown away when the test finishes.

Run locally:  python tests/smoke_test.py
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def check_imports():
    """Every platform package must import cleanly. Importing api.app also pulls in
    common/, adapters/, compressors/, export/, registry/ and FastAPI in one shot."""
    import adapters          # noqa: F401
    import common            # noqa: F401
    import compressors       # noqa: F401
    from common.hardware import compressors_for   # noqa: F401
    from export import list_exporters             # noqa: F401
    from registry import mlflow_state, promote    # noqa: F401
    import api.app           # noqa: F401
    print("[ok] package imports")


def check_schemas():
    import jsonschema
    from jsonschema import Draft7Validator
    for rel in ("plugin/job.schema.json", "report/report.schema.json"):
        schema = json.load(open(os.path.join(REPO, rel)))
        Draft7Validator.check_schema(schema)
    print("[ok] JSON schemas valid")


def check_engine(tmp: str):
    """Generate a tiny fixture model + test set, run the plugin end to end, and
    assert a schema-valid compression report comes out."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (300, 4)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model = LogisticRegression(max_iter=200).fit(X, y)

    model_path = os.path.join(tmp, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["label"] = y
    test_path = os.path.join(tmp, "test.csv")
    df.to_csv(test_path, index=False)

    job = {
        "model": {"name": "smoke", "ref": model_path, "framework": "sklearn",
                  "task": "binary_classification", "target": "label"},
        "test_data": {"ref": test_path},
        "target_hardware": "cpu-generic",
    }
    job_file = os.path.join(tmp, "job.json")
    with open(job_file, "w") as f:
        json.dump(job, f)

    art_dir = os.path.join(tmp, "artifacts")
    subprocess.run(
        [sys.executable, os.path.join(REPO, "plugin", "run_job.py"),
         "--job", job_file, "--artifacts-dir", art_dir],
        cwd=REPO, check=False,
    )

    report_path = os.path.join(art_dir, "smoke", "compression_report.json")
    assert os.path.exists(report_path), f"engine produced no report at {report_path}"

    report = json.load(open(report_path))
    schema = json.load(open(os.path.join(REPO, "report", "report.schema.json")))
    import jsonschema
    jsonschema.validate(report, schema)
    assert isinstance(report["gate_passed"], bool)
    print(f"[ok] engine ran: best={report['best_variant']['name']} "
          f"size={report['deltas']['size_delta_pct']}% gate={report['gate_passed']}")


def main():
    check_imports()
    check_schemas()
    with tempfile.TemporaryDirectory() as tmp:
        check_engine(tmp)
    print("\nPLATFORM SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
