# MLOps Pipeline for Model Compression

A framework-agnostic pipeline that takes **any number of** trained models,
compresses each every applicable way, benchmarks the variants, gates on accuracy,
registers the winners to MLflow, and auto-deploys them to a **free Hugging Face
Space**.

Originally XGBoost-churn-only, now driven entirely by [`pipeline.yaml`](pipeline.yaml)
and multi-model (compresses a whole list of unrelated models in one run).

```
train ─▶ compress ─▶ benchmark ─▶ gate ─▶ register ─▶ deploy ─▶ monitor
        (any model → ONNX FP32 → INT8 variants)     (MLflow)   (HF Spaces)
```

## Supported frameworks & compression

| Framework | Export | Compression that runs |
|---|---|---|
| XGBoost / LightGBM / scikit-learn | ONNX (onnxmltools / skl2onnx) | ONNX INT8 dynamic ⭐, INT8 static, TRT INT8 (GPU) |
| PyTorch | ONNX (torch.onnx) | ONNX INT8 dynamic ⭐, INT8 static, TRT INT8 (GPU) |

> **No GPU needed.** ONNX-Runtime INT8 quantization gives real compression on
> CPU. TensorRT is attempted only if an NVIDIA GPU is present and falls back to
> FP32 ONNX otherwise.

## Architecture (the generic core)

- **`pipeline.yaml`** — single source of truth: framework, task, feature schema,
  compressors to run, gate thresholds, deploy target. No more duplicated feature lists.
- **`common/`** — config loader, feature schema → dynamic Pydantic request model,
  task-aware metrics, uniform ONNX output parsing.
- **`adapters/`** — one class per framework: `load`, `predict`, `to_onnx`.
  Register a new framework by adding an adapter to `adapters/__init__.py`.
- **`compressors/`** — one class per method producing a `Variant`. Add a technique
  (e.g. pruning, distillation) by dropping a module in and listing it in the registry.
- **`artifacts/variants.json`** — the manifest every downstream stage consumes.

## Run locally (Windows)

```bat
run_pipeline.bat
```

First run auto-installs `requirements.txt`, starts MLflow (`python -m mlflow server`),
generates data, trains, compresses, benchmarks, gates, registers, deploys, and
launches the API.

- API + Swagger: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

Manual install (if you prefer): `python -m pip install -r requirements.txt`
(uncomment `lightgbm` / `torch` in the file for those frameworks).

## Point it at one or many models

`pipeline.yaml` has a `models:` list. Add as many as you want — they can be
totally unrelated (different frameworks, tasks, features, datasets). Every stage
loops over them and compresses each independently into `artifacts/<name>/`, indexed
by `artifacts/models.json`.

```yaml
models:
  - name: churn
    framework: xgboost
    task: binary_classification
    artifact: artifacts/churn/model.pkl
    trainable: true                 # built-in trainer produces this
    target: churn
    data: { train: data/train.csv, test: data/test.csv }
    features: [ ... ]

  - name: house_price               # a second, unrelated model
    framework: sklearn
    task: regression
    artifact: artifacts/house_price/model.pkl
    trainable: false                # you drop in the trained artifact
    target: price
    data: { test: data/house_test.csv }
    features: [ ... ]
```

- **Trainable classic-ML models** (`trainable: true`, framework `xgboost`): the
  built-in trainer produces the artifact.
- **Everything else** (PyTorch, sklearn, LightGBM you trained yourself): set
  `trainable: false` and drop your trained artifact at `model.artifact`.

### Multi-model serving

One API serves every model, each with a schema built from its own feature list:

```
POST /predict/{model}/{variant}     e.g. /predict/house_price/onnx_fp32
POST /compare/{model}               runs all of that model's variants
GET  /models                        lists models, tasks, variants, features
```

> **Chained models (A→B):** the pipeline compresses each model independently. If
> you need it to *run* model A and feed its outputs into model B for benchmarking,
> that's an inference-topology add-on — not built yet.

## Deploy to Hugging Face Spaces (free, no EC2)

1. Create a token at https://huggingface.co/settings/tokens (write scope).
2. Set `deploy.space_id: "your-username/your-space"` in `pipeline.yaml`.
3. Export the token and run the deploy step:

   ```bat
   set HF_TOKEN=hf_xxx
   python deploy\hf_spaces_deploy.py
   ```

It creates a Docker Space, uploads the serving code + compressed artifacts, and HF
builds a public API. The step is skipped automatically if `HF_TOKEN`/`space_id`
are unset, so `run_pipeline.bat` never fails on it.

## Adding a new compression technique

Implement `compressors.base.Compressor.compress(...)` returning a `Variant`,
register it in `compressors/__init__.py`, and add its name to `compression.methods`
in `pipeline.yaml`. Benchmark/gate/serve pick it up with no other changes.
