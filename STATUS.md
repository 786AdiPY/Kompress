# Kompress — Project Status

_Last updated: 2026-08-30_

## What Kompress is

A **model-compression platform** (not just a pipeline). You give it a trained model +
a test set + the hardware it will deploy to; it compresses the model every applicable
way, benchmarks the variants, gates on accuracy, produces a **"Plan"** (size/latency/
accuracy deltas), stores the winner in **MLflow**, and — on human consent — promotes it
to Production and lets you export it for the target device.

It has **two front doors onto one engine**:

| | **Front Door A** — orchestrator plugin | **Front Door B** — self-serve platform |
|---|---|---|
| Who uses it | Jenkins / GitHub Actions / Airflow | A human, via the dashboard |
| Entry point | `src/kompress/services/plugin/run_job.py` (`kompress-job`) | `src/kompress/services/cloud/api/app.py` (`kompress-api`) + `web/` (React UI) |
| Where compute runs | the caller's own runner | our workers (queue + worker pool) |
| MLflow | the caller's own (or none) | the platform's own |

The unit of work is the same everywhere: `src/kompress/services/plugin/run_job.py` running the engine image.

## Architecture / component map

```
src/kompress/
├── engine/
│   ├── adapters/      framework plugins: xgboost | lightgbm | sklearn | pytorch -> ONNX
│   ├── compressors/   technique plugins: onnx_fp32 | onnx_int8_dynamic/static | trt_int8
│   ├── export/        device export targets: onnx (✓) | tensorrt (✓) | tflite (stub) | coreml (stub)
│   ├── pipeline/      benchmark.py, compress.py, gate.py, generate_report.py (the "Plan")
│   ├── registry/      mlflow_state.py (tag-based approval state) + promote.py (declarative promote/rollback)
│   └── common/        config loader, feature schema, metrics, hardware map, objstore, metrics_store
├── services/
│   ├── cloud/api/     FastAPI (Front Door B): submit, review queue, approve/reject/rollback, export, uploads
│   ├── cloud/worker/  executor.py (run job -> store in MLflow & Supabase) + queue.py + worker.py
│   ├── cloud/web/     React + TS dashboard: Landing, Dashboard, Submit (file upload UI), Run Detail, Deployments
│   ├── cloud/serve/   Model serving endpoint wrapper
│   └── plugin/        job manifest schema + run_job.py (Front Door A entry point)
└── tools/             train.py, monitor.py, hf_deploy.py
tests/                 smoke_test.py (platform CI gate)
.github/workflows/ci.yml   platform CI (backend smoke + frontend build)
.github/actions/compress-model   reusable GitHub Action (Front Door A adapter)
integrations/README.md     how orchestrators call Front Door A
```

### The state machine lives in MLflow tags (no separate DB)
`pending_gate → pending_approval → approved | rejected | failed`. The dashboard's review
queue is just `list_by_tag(status=...)`; approve/reject/rollback set the tag. See
`src/kompress/engine/registry/mlflow_state.py`.

### Execution model (Front Door B)
`POST /runs` creates an MLflow run (`pending_gate`) and **enqueues** a job. A separate
**worker** (`src/kompress/services/cloud/worker/worker.py`) claims it, runs the engine in isolation, and **stores the
result in MLflow** (delta metrics + registered ONNX + status tag). Toggle with
`KOMPRESS_EXECUTION=inline|queue`. Queue backend is a dependency-free filesystem queue
(`src/kompress/services/cloud/worker/queue.py`); Redis/RQ or one Kubernetes Job per message swap in via the same interface.

## What works / verified

- ✅ **Package Restructure & Packaging** — Restructured codebase into an installable `kompress` Python package under `src/kompress/` with `pyproject.toml` console scripts (`kompress-job`, `kompress-api`, `kompress-worker`).
- ✅ **Storage & Database Connectors** — Cloudflare R2 / Backblaze B2 object storage integration via `src/kompress/engine/common/objstore.py` and Supabase Postgres run metrics logging (`run_metrics` table) via `src/kompress/engine/common/metrics_store.py`.
- ✅ **Direct File Upload UI** — Integrated direct file upload intake on `SubmitJob.tsx` (`POST /uploads`) streaming `.pkl` models & `.csv` test datasets directly to object storage and passing generated `s3://` pointers to `POST /runs`.
- ✅ **Front Door A** — job manifest → materialized pipeline → compress/benchmark/gate/report; hardware drives compressor selection; features auto-inferred from the test CSV.
- ✅ **Report ("Plan")** — schema-validated `compression_report.json` (base hash, size/latency/accuracy deltas, variants table, gate result).
- ✅ **Front Door B API** — submit, uploads, review queue, report, artifact download, approve → promote, reject, rollback, `/models`, `/export`.
- ✅ **Approval → MLflow** — approve registers the winning ONNX to the Model Registry and promotes to Production; rollback = promote an older run.
- ✅ **Worker/queue** — verified end-to-end: submit → queued → worker compresses → **registered in MLflow & Supabase** → `pending_approval`.
- ✅ **Streamlined UI Compression Flow** — Refactored Run Detail to follow a linear 5-step flow: `Compression Results` → `Plan Overview` → `Accuracy Gate: PASSED` → `Recommended Variant (e.g. INT8)` → `[ Download / Export ]`. Scoped the Deployments view to automated CI pipeline deployment status.
- ✅ **HuggingFace Deployment Tooling** — Added `src/kompress/tools/hf_deploy.py` for automated HF Spaces deployment.
- ✅ **Platform CI** — `tests/smoke_test.py` (imports + schemas + engine on a fixture) verified and passing with package structure; frontend `npm run build` passes.
- ✅ **Dashboard** — Landing + 4 pages build clean; API contract review passed (`contract_matches: true`).
- ✅ **Cleaned Git Tracking** — Purged unneeded generated MLflow databases and output artifacts from git tracking, keeping working tree clean.

## Currently working on / pending

- ⏳ **Postgres & R2 Integration Branch** — Branch `feature/postgres-r2-integration` active with Supabase and Cloudflare R2 / B2 storage connectors wired into the pipeline and upload handler.
- ⏳ **DL / PyTorch path** — newer `torch` (2.13) changed `torch.onnx.export` (dynamo default); `src/kompress/engine/adapters/pytorch_adapter.py` needs a `dynamo=False` fix for the DL sample to compress. Not in CI (smoke test uses sklearn) and `torch` is opt-in, so it doesn't affect the build.
- ✅ **Design** — product-grade UI refresh (Lexend, design tokens, product tour) and marketing landing page landed.

## Sample models to test with

| | Churn (traditional ML) | House Price (traditional ML) | Churn (DL) |
|---|---|---|---|
| Model ref | `artifacts/churn/model.pkl` | `artifacts/house_price/model.pkl` | `artifacts/churn_torch/model.pt` |
| Test ref | `data/test.csv` | `data/house_test.csv` | `data/test.csv` |
| Framework / Task | xgboost / binary_classification | sklearn / regression | pytorch / binary_classification |
| Target | `churn` | `price` | `churn` |
| Status | ✅ works | ✅ works | ⏳ needs the torch 2.13 adapter fix |

Leave **Features blank** in the UI — they're auto-inferred from the test CSV (every column except the target). **Set Target** or the target column gets treated as a feature.

## How to run it (Linux / Arch)

```bash
./run.sh              # sets up venv, installs editable kompress package, starts everything with live auto-reload
./run.sh --with-dl    # also installs torch + onnxscript for the PyTorch path
```

> **Live Auto-Reloading**: You do **not** need to restart `./run.sh` on code updates.
> - **React UI (`web/`)**: Vite Dev Server automatically hot-reloads (HMR) browser UI changes instantly upon saving.
> - **Python Backend (`src/kompress/`)**: Installed in editable mode (`pip install -e .`) with Uvicorn watcher (`--reload --reload-dir src`), auto-reloading API endpoints dynamically on every Python code change.

Services: **Dashboard** http://localhost:5173 · **API** http://localhost:8000/docs ·
**MLflow UI** http://localhost:5000 (best-effort; needs Python < 3.14). Ctrl+C stops all.

## How to test end-to-end

1. `./run.sh` → open the **Dashboard**.
2. **Submit** a job with the churn sample (table above).
3. Watch the run go `pending_gate → pending_approval` (the worker picks it up in a few seconds).
4. Open it → review the **Plan** → **Approve** → it promotes to Production; see it in **Deployments** and the **MLflow UI**.
5. To prove the queue decoupling: stop the worker, submit — the job sits in `queue/pending/` and status stays `pending_gate` until a worker runs.

## Testing your OWN (external) models

You compress a model you bring — the bundled churn/house samples are just conveniences.
A job needs three things: the **model**, its **test set**, and the **target hardware**
(framework/task/target come along too; features are auto-inferred from the test CSV).

### Size tiers: direct upload (<10GB, primary) vs. pointer-only (≥10GB, deferred)

Two intake paths, chosen automatically by size — not a UI toggle:

| Tier | Size | Path | Status |
|---|---|---|---|
| **Direct upload** | **< 10GB** (`KOMPRESS_MAX_UPLOAD_GB`, default 10) | `POST /uploads` streams the file into object storage (MinIO/S3), returns an `s3://` pointer, submit that to `POST /runs` | ✅ primary path, implemented |
| **Pointer-only** | **≥ 10GB** | caller hosts the file themselves and submits its `s3://…` ref directly to `POST /runs` — `/uploads` refuses it | supported, not the tuned/tested path right now (deferred) |

`POST /uploads` enforces the ceiling **before** doing any real work: a `Content-Length` over the
limit is rejected immediately (`413`, no upload attempted); a chunked/lying upload is aborted
**mid-stream** the moment it crosses the limit (via a size-counting wrapper around the read —
verified: an oversized stream is caught after a few chunks, not after buffering the whole file).
Get a `413` → the message tells you to submit a pointer instead.

Why pointer, not always-upload: an uploaded pickle is **arbitrary code execution** on unpickle;
some datasets are legitimately huge; and pointers keep the API stateless. Below 10GB we accept the
UX cost of "upload a file" by re-hosting it ourselves (object storage) so the engine still only
ever consumes pointers internally — the raw bytes never reach the compression process.

**Resource sizing for the upload tier** (`deploy/docker-compose.oss.yml`): workers are sized
`mem_limit: 32g` / `cpus: 4` per the rule of thumb in `src/kompress/engine/common/limits.py` — ~2-3× model size in
memory (load + export), ~3-4× in scratch disk (original + ONNX export + variants + calibration
data) → ~30GB RAM / ~40GB disk headroom per concurrent job at the 10GB ceiling. `deploy: replicas: 2`
means 2 concurrent jobs at that sizing; scale workers and this budget multiplies per replica.
nginx (`deploy/nginx.conf`) is set to `client_max_body_size 11g` (headroom above the 10GB app-level
cap, so the API's own 413 fires — not nginx's generic one) with 1800s timeouts for the transfer.

### How to test it

**A. Local (`./run.sh`, `API_ALLOW_LOCAL_PATHS=1`)** — drop your files in the repo and reference them
by path (no object store needed, and no size ceiling applied — that check only lives on `/uploads`):

```bash
# put your model + test csv anywhere under the repo, then submit:
curl -X POST localhost:8000/runs -H 'content-type: application/json' -d '{
  "model":{"name":"my_model","ref":"path/to/my_model.pkl","framework":"sklearn",
           "task":"regression","target":"price"},
  "test_data":{"ref":"path/to/my_test.csv"},
  "target_hardware":"cpu-generic"}'
```

**B. OSS stack / hosted (`API_ALLOW_LOCAL_PATHS=0`)** — upload to object storage first (this is
exactly what an "upload a file" button in the UI does under the hood), then submit the pointer:

```bash
# 1. upload the model -> get an s3:// pointer (413 if it's >= the 10GB ceiling)
curl -F 'file=@my_model.pkl' -F 'kind=model' localhost:8000/uploads
#    -> {"ref":"s3://kompress/uploads/model/…_my_model.pkl", …}
# 2. upload the test set the same way (kind=test_data), then
# 3. POST /runs with those two refs.
```

Requirements for the model you bring:
- **Framework** one of `xgboost | lightgbm | sklearn | pytorch`; **task** one of
  `binary_classification | multiclass_classification | regression`.
- The **test CSV** has one column per feature plus the **target** column — set `target` correctly
  or it gets treated as a feature. Leave `features` empty to auto-infer.
- **PyTorch** models: save as TorchScript (`torch.jit.save`) or a whole-module `torch.save`; the DL
  path also needs the `--with-dl` deps (currently pending the torch-2.13 `dynamo=False` adapter fix).

The run then flows like any other: `pending_gate → pending_approval` (worker compresses it) →
review the Plan → **Approve** → stored & promoted in MLflow.

## Environment notes

- **Python 3.14 + MLflow server**: the MLflow *UI server* crashes on 3.14 (`importlib.abc.Traversable` was removed). The platform itself (API + worker) works on any Python; `run.sh` starts the UI best-effort and falls back to a direct sqlite store if the server won't start.
- Runtime dirs (`api_runs/`, `queue/`, `mlruns/`, `mlartifacts/`, `*.db`) and `web/node_modules`, `web/dist` are git-ignored.

---

## Deploying Kompress in the cloud (and exposing its API for orchestration)

The goal: host Kompress so that **orchestrators (Jenkins / GitHub Actions / Airflow / Argo)
call its API over the network** to compress models — the SaaS/hosted form of Front Door A.

> Note on the two ways to consume Kompress:
> - **Plugin (no hosting needed):** the caller runs the engine image in their own CI
>   (`docker run … python src/kompress/services/plugin/run_job.py`). Nothing of ours in the cloud.
> - **Hosted API (this section):** the caller hits our cloud endpoint. This is what
>   "provide its API for orchestration" means.

### Target topology

```
   Orchestrator (Jenkins/Airflow/GH Action)
        │  HTTPS + API key
        ▼
   ┌──────────────┐        ┌───────────────────────────┐
   │  Ingress /   │──────▶ │  API (FastAPI)  Deployment │  stateless, N replicas, HPA
   │  LoadBalancer│        │  .../cloud/api/app.py     │  KOMPRESS_EXECUTION=queue
   └──────────────┘        └───────────┬───────────────┘
        │ (static)                     │ enqueue
        ▼                              ▼
   ┌──────────────┐        ┌───────────────────────────┐
   │  Dashboard   │        │  Queue  (Redis / SQS /     │
   │  (CDN/bucket)│        │  PubSub, or K8s Jobs)      │
   └──────────────┘        └───────────┬───────────────┘
                                        │ claim
                                        ▼
                           ┌───────────────────────────┐
                           │  Workers (...worker.py)   │  scale on queue depth (KEDA);
                           │  1 isolated pod/Job per run│  sandboxed — pickle = RCE risk
                           └───────────┬───────────────┘
                                        │ store
                                        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  MLflow  ── tracking server + Postgres + object store (S3/GCS) │  system of record
   └──────────────────────────────────────────────────────────────┘
   Model & data pointers (s3://, gs://, mlflow://) live in the SAME object store.
```

### Component-by-component

| Component | Cloud form | Notes |
|---|---|---|
| **API** | container Deployment behind an Ingress/LB (K8s), or Cloud Run / ECS Fargate / Container Apps | stateless (state is in MLflow); set `KOMPRESS_EXECUTION=queue`; autoscale on CPU/RPS |
| **Workers** | separate Deployment, **or one Kubernetes Job per message** (best isolation) | scale on **queue depth** (KEDA / HPA); each pod ephemeral & sandboxed |
| **Queue** | Redis (managed), SQS, Pub/Sub — **or** native K8s Jobs (API creates a Job, no broker) | swap `src/kompress/services/cloud/worker/queue.py`'s `FileQueue` for a `RedisQueue` (same interface) |
| **MLflow** | tracking server container + **managed Postgres** + **S3/GCS artifact store** | replaces the dev sqlite; needed for concurrent writers + durable artifacts |
| **Dashboard** | `npm run build` → static assets on a bucket + CDN, or an nginx sidecar | point it at the API base URL (`VITE_API_BASE`) |
| **Object storage** | S3 / GCS / Azure Blob | holds uploaded model + test-data pointers AND the MLflow registry artifacts |
| **Secrets/Config** | K8s Secrets / cloud secret manager | everything is env-driven already (see below) |

### What must be hardened before production (honest gaps)

The current build is dev-grade. To expose the API publicly you need to add:

1. **AuthN/AuthZ** — the API has **no auth today**. Add an API-key or OAuth2/JWT dependency
   (FastAPI dependency on every route) + per-tenant keys. Without this, do **not** expose it.
2. **Queue backend** — `FileQueue` needs a shared filesystem (fine on one node / a shared PVC,
   not across nodes). Implement `RedisQueue(QueueBackend)` in `src/kompress/services/cloud/worker/queue.py`, or have the API
   create a **K8s Job per run** and drop the broker entirely.
3. **MLflow store** — move off sqlite to **Postgres + object-store artifacts**
   (`--backend-store-uri postgresql://… --default-artifact-root s3://…`).
4. **Multi-tenancy & quotas** — one MLflow experiment (or registry prefix) per tenant; run/CPU quotas.
5. **Worker sandboxing** — an uploaded pickle is arbitrary code execution. Run workers as isolated,
   least-privilege, network-restricted pods (no cloud creds beyond the one artifact bucket).
6. **Pointer fetchers** — `_fetch_s3` (boto3) exists; add GCS/Azure + presigned-URL support and give
   the worker read-only creds for the data bucket.

### How an orchestrator consumes the hosted API

```
# 1. submit — pointers only (S3/GCS/MLflow URI), never file uploads
POST /runs                Authorization: Bearer <api-key>
  { "model": {"ref":"s3://bucket/model.pkl","framework":"xgboost","task":"binary_classification","target":"churn"},
    "test_data": {"ref":"s3://bucket/test.csv"},
    "target_hardware": "nvidia-gpu" }
  → 202 { "run_id": "…", "status": "pending_gate" }

# 2. poll (or receive a webhook) until terminal
GET  /runs/{run_id}        → { "status": "pending_approval" | "rejected" | "failed" }

# 3. read the Plan and decide
GET  /runs/{run_id}/report → compression_report.json (size/latency/accuracy deltas, gate)

# 4a. in CI you typically auto-approve on gate pass -> promotes in the MLflow registry
POST /runs/{run_id}/approve
# 4b. or pull the compressed model to deploy yourself (local / IoT / mobile)
GET  /runs/{run_id}/export?format=onnx|tensorrt|tflite|coreml
```

A ready-made GitHub Action wrapper lives at `.github/actions/compress-model`; a Jenkins/Airflow
step is just this same HTTP sequence. See `integrations/README.md`.

### Deployment options (pick per scale)

- **Kubernetes (recommended for MLOps fit):** API Deployment + Ingress(TLS); workers as a
  Deployment (KEDA-scaled on queue length) **or** a Job-per-run controller; MLflow Deployment +
  managed Postgres + S3/GCS; dashboard on a bucket+CDN; secrets via K8s Secrets. Autoscaling and
  per-job isolation come for free.
- **Managed containers (simplest to stand up):** API + worker on Cloud Run / ECS Fargate /
  Container Apps; queue = SQS / Pub/Sub; MLflow on a small instance + RDS/CloudSQL + object store;
  dashboard on static hosting + CDN.
- **Single VM (dev / pilot):** everything via docker-compose on one box with `FileQueue` on a
  shared volume — mirrors `./run.sh`, not horizontally scalable.

### Open-source / self-hosted alternatives (no vendor lock-in)

Kompress can run **entirely on open-source infrastructure** — every managed service above
has a self-hostable OSS equivalent, and the core (MLflow + the compression libraries) is
already OSS. This matters for on-prem, air-gapped, and cost-sensitive deployments, and is a
differentiator against closed compression platforms.

| Need | Managed option | Open-source, self-hostable |
|---|---|---|
| Queue | SQS / Pub/Sub / managed Redis | **Valkey** (OSS Redis fork), **RabbitMQ**, **NATS**, or **K8s Jobs** (no broker) |
| Object store | S3 / GCS / Azure Blob | **MinIO** (S3-compatible), SeaweedFS, Ceph |
| Database (MLflow backend) | RDS / CloudSQL | **PostgreSQL**, MariaDB |
| Container platform | ECS / Cloud Run / Container Apps | **Kubernetes** (k3s / kind / MicroK8s), Nomad, Docker Swarm |
| Autoscale on queue depth | cloud autoscalers | **KEDA** (CNCF) |
| Ingress / load balancer | ALB / Cloud LB | **Traefik**, ingress-nginx, HAProxy, **MetalLB** (bare metal) |
| Auth / API gateway | Cognito / API Gateway | **Keycloak** / Authentik / Ory (OIDC), **Kong** / APISIX / Tyk |
| Secrets | Secrets Manager / KMS | **HashiCorp Vault**, Sealed Secrets, SOPS |
| Observability | CloudWatch | **Prometheus + Grafana + Loki + OpenTelemetry** |
| Experiment/model registry | SageMaker / Vertex registry | **MLflow** — already what Kompress uses (Apache-2.0) |
| DAG orchestration (optional) | Step Functions / managed Airflow | **Argo Workflows**, **Apache Airflow**, Kubeflow Pipelines |

**The compression engine itself is already fully open source:** ONNX Runtime (MIT),
skl2onnx / onnxmltools (Apache/MIT), and **Sony MCT** (Apache-2.0) for the edge-NPU path.
The only proprietary dependency anywhere is **TensorRT** (NVIDIA) — and it is optional
(the pipeline falls back to ONNX-Runtime INT8 without a GPU).

A fully-OSS reference stack:
**k3s** (orchestration) · **MinIO** (artifacts) · **PostgreSQL** (MLflow backend) ·
**Valkey** or **K8s Jobs** (queue) · **KEDA** (worker autoscaling) · **Keycloak** (API auth) ·
**Traefik** (ingress) · **Prometheus/Grafana/Loki** (observability) · **MLflow** (registry).

### Connectors — what each one does (wired in `deploy/docker-compose.oss.yml`)

Kompress is deployment-ready with the OSS stack via pluggable, env-driven connectors:

| Connector | Code / selector | What it does |
|---|---|---|
| **FileQueue** (default) | `src/kompress/services/cloud/worker/queue.py`, `KOMPRESS_QUEUE=file` | zero-dependency filesystem job queue for single-node / dev |
| **RedisQueue** | `src/kompress/services/cloud/worker/queue.py`, `KOMPRESS_QUEUE=redis://…` (or `valkey://`) | multi-node job queue on **Valkey/Redis**; atomic claim via `LMOVE` (reliable-queue: a crashed worker's job isn't lost) |
| **Object store** | `src/kompress/engine/common/objstore.py`, `S3_ENDPOINT_URL` + `AWS_*` | **MinIO/S3** — resolves `s3://` pointers (download) and backs uploads. Endpoint-aware: set `S3_ENDPOINT_URL=http://minio:9000` for MinIO, unset for real AWS S3 |
| **Upload endpoint** | `POST /uploads` (`src/kompress/services/cloud/api/app.py`) | streams a user's uploaded `.pkl`/CSV into object storage and returns an `s3://` pointer — so the UI offers "upload a file" while the engine still only ever consumes pointers. Enforces the **<10GB direct-upload ceiling** (`src/kompress/engine/common/limits.py`, `KOMPRESS_MAX_UPLOAD_GB`) — `413` fast-path on `Content-Length`, aborts mid-stream otherwise; ≥10GB models use the pointer-only path directly against `POST /runs` instead |
| **MLflow (Postgres + MinIO)** | `MLFLOW_TRACKING_URI`, `deploy/Dockerfile.mlflow` | tracking server + model registry; **PostgreSQL** metadata (concurrent-writer safe) + **MinIO** artifact store (replaces dev sqlite) |
| **Dashboard (nginx)** | `web/Dockerfile`, `deploy/nginx.conf` | serves the built SPA and reverse-proxies `/api` → the API service |
| **OSS compose** | `deploy/docker-compose.oss.yml` | one command wires Postgres + MinIO + Valkey + MLflow + API + N workers + dashboard |

**Deploy the whole OSS stack:**

```bash
docker compose -f deploy/docker-compose.oss.yml up --build
#   Dashboard  http://localhost:8080
#   API + docs http://localhost:8000/docs
#   MLflow UI  http://localhost:5000
#   MinIO console http://localhost:9001   (minioadmin / minioadmin)
#
# scale workers:  docker compose -f deploy/docker-compose.oss.yml up --scale worker=4
```

> Auth is intentionally **not** wired yet — put an API gateway (Kong/APISIX) or add an
> API-key/OIDC dependency in front before exposing the API publicly.

### Config reference (all env-driven — 12-factor)

| Var | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | the platform's MLflow (Postgres-backed server URL in prod) |
| `MLFLOW_EXPERIMENT` | experiment / tenant namespace |
| `KOMPRESS_EXECUTION` | `queue` in prod (API only enqueues) |
| `KOMPRESS_QUEUE` / `KOMPRESS_QUEUE_DIR` | queue backend selector / FileQueue path |
| `API_RUNS_DIR` | per-run working dir (a shared PVC in K8s) |
| `API_ALLOW_LOCAL_PATHS` | **keep `0` in prod** — pointers only, no local paths |
| `CORS_ORIGINS` | the dashboard's origin(s) |
| `PROJECT` | registered-model name prefix |

**Bottom line:** the code is already structured for this — stateless API, MLflow-as-state,
a pluggable queue, and an isolated worker unit. The cloud lift is **swap sqlite→Postgres/S3,
FileQueue→Redis (or K8s Jobs), add auth, and containerize** — not a rewrite.
