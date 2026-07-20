# Integrations — using the compression platform from your MLOps stack

The platform has **one core engine** and **two front doors** onto it. Everything
here is Front Door A: how an external orchestrator runs compression as a stage.

```
                    ┌─────────────────────────────────────────────┐
                    │  Core engine                                 │
                    │  plugin/run_job.py  (job manifest in,        │
                    │  compressed variant + compression_report.json out)
                    └─────────────────────────────────────────────┘
                       ▲                                   ▲
        Front Door A   │                                   │  Front Door B
   (this doc: headless,│                                   │  (api/ — humans,
    any orchestrator)  │                                   │   consent, dashboard)
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
  truth. Front Door A never depends on the platform's infra.

## Adapter layers (pick the thinnest that fits)

| Layer | Use when | How |
|---|---|---|
| Raw `docker run` | Any shell-capable runner (GitLab CI, Argo, cron) | Call the command above in a script step. |
| **GitHub Action** (provided) | GitHub Actions | `uses: ./.github/actions/compress-model` — see below. |
| Jenkins shared-library step | Jenkins pipelines | Wrap the `docker run` in a `compressModel(job)` step (thin, on the roadmap). |
| Airflow / Argo operator | You need DAG-native retries/XCom | Defer until a concrete need — the operator just shells to the same command. |

Recommendation: **start with raw `docker run` or the GitHub Action.** Build a
native Airflow/Argo operator only when a user actually needs DAG-level wiring —
all of them ultimately call the identical `run_job.py` command, so there is no
capability gained by building them early, only maintenance.

## GitHub Actions (reference adapter)

```yaml
jobs:
  compress:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - id: compress
        uses: ./.github/actions/compress-model
        with:
          job: plugin/job.example.yaml
          artifacts-dir: artifacts
          # mlflow-tracking-uri: http://your-mlflow:5000   # optional
      - run: echo "Gate passed: ${{ steps.compress.outputs.gate-passed }}"
```

The action writes a **job summary table** (model, best variant, size Δ, latency Δ,
gate) to the run, and fails the step on a gate failure so it blocks a merge/deploy
like any other check. See [`.github/actions/compress-model/action.yml`](../.github/actions/compress-model/action.yml).

## Jenkins (pattern)

```groovy
stage('Compress') {
  steps {
    sh '''
      docker run --rm -v "$WORKSPACE:/work" -w /work compression-pipeline:local \
        python plugin/run_job.py --job job.yaml --artifacts-dir artifacts
    '''
    archiveArtifacts artifacts: 'artifacts/**/compression_report.json'
  }
}
```

For a human approval gate *inside* Jenkins, wrap the promotion (not the
compression) in an `input` step that calls
[`registry/promote.py`](../registry/promote.py) `--run-id <id>` — the same
declarative promote/rollback the self-serve UI uses.
