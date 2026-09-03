# Integrations — using the compression platform from your MLOps stack

The platform has **one core engine** and **two entry gates** onto it. Everything
here is Gate B: how an external CLI or orchestrator runs compression as a stage.

```
                    ┌─────────────────────────────────────────────┐
                    │  Core engine                                │
                    │  plugin/run_job.py  (job manifest in,       │
                    │  compressed variant + compression_report.json out)
                    └─────────────────────────────────────────────┘
                       ▲                                   ▲
        Gate B         │                                   │  Gate A
   (this doc: headless,│                                   │  (cloud / api /
    orchestration/CLI) │                                   │   consent, dashboard)
```

## The one command every orchestrator calls

There is a single entrypoint. Adapters differ only in how they *invoke* it:

```bash
python plugin/run_job.py --job <job.yaml> --artifacts-dir <out>
# or, containerized (recommended — pinned deps, no host setup):
docker run --rm -v "$PWD:/work" -w /work compression-pipeline:local \
    python plugin/run_job.py --job job.yaml --artifacts-dir artifacts
```

- **Input**: a job manifest — `{model, test_data, target_hardware, gate}` — see
  [`plugin/job.schema.json`](../plugin/job.schema.json) and
  [`plugin/job.example.yaml`](../plugin/job.example.yaml). Test set only; no training data.
- **Output**: `artifacts/<model>/compression_report.json` (the "Plan") + the
  compressed variants. Exit code is non-zero iff the accuracy gate fails.
- **MLflow is optional here.** Set `MLFLOW_TRACKING_URI` to *your own* tracking
  server to log the run; leave it unset and the report file is the source of
  truth. Gate B never depends on the platform's infra.

## Adapter layers (pick the thinnest that fits)

| Layer | Use when | How |
|---|---|---|
| **Raw CLI / Bash** | Jenkins, Airflow `BashOperator`, local terminal | `python plugin/run_job.py --job job.yaml` |
| **Docker** | Containerized pipelines | `docker run --rm -v $PWD:/work ...` |
| **GitHub Action** | GitHub Actions workflows | `uses: ./.github/actions/compress-model` |

## GitHub Actions

Use the reusable action in your workflow:

```yaml
steps:
  - uses: actions/checkout@v4
  - name: Compress & Gate Model
    uses: ./.github/actions/compress-model
    with:
      job-manifest: 'plugin/job.example.yaml'
```

## Airflow / Prefect / Dagster

In Airflow, call `run_job.py` via `BashOperator` or `DockerOperator`:

```python
from airflow.operators.bash import BashOperator

compress_task = BashOperator(
    task_id="compress_model",
    bash_command="python plugin/run_job.py --job /path/to/job.yaml --artifacts-dir /path/to/artifacts",
)
```
