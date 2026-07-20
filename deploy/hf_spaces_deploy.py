"""Auto-deploy the compressed model API to a Hugging Face Space (Docker SDK).

Runs after the gate passes. Packages serve/ + the pipeline core + the compressed
artifacts into a Docker Space and pushes it. The Space builds and serves the same
FastAPI app on HF's free tier — a public URL, no EC2.

Env:
  HF_TOKEN   — a Hugging Face token with write scope (required)
  SPACE_ID   — "username/space-name" (falls back to deploy.space_id in config)
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config  # noqa: E402

MODELS_INDEX = os.getenv("MODELS_INDEX", "artifacts/models.json")

# Serving-only deps; framework libs added below based on the manifest.
BASE_REQS = [
    "fastapi>=0.111.0", "uvicorn[standard]>=0.29.0", "pydantic>=2.7.1",
    "onnxruntime>=1.18.0", "numpy>=1.26.4", "pyyaml>=6.0",
]
FRAMEWORK_REQS = {
    "xgboost":  ["xgboost>=2.0.3", "scikit-learn>=1.4.2"],
    "lightgbm": ["lightgbm>=4.0.0", "scikit-learn>=1.4.2"],
    "sklearn":  ["scikit-learn>=1.4.2"],
    "pytorch":  ["torch>=2.2.0"],
}

DOCKERFILE = """\
FROM python:3.11-slim
WORKDIR /app
COPY spaces_requirements.txt .
RUN pip install --no-cache-dir -r spaces_requirements.txt
COPY . .
ENV MODELS_INDEX=artifacts/models.json PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""


def readme(project, framework, task, space_id):
    return f"""---
title: {project} Compression API
emoji: 🗜️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# {project} — Model Compression API

Auto-deployed by the MLOps compression pipeline.

- **Framework:** {framework}
- **Task:** {task}
- **Endpoints:** `POST /predict/{{variant}}`, `POST /compare`, `GET /models`, `GET /health`
- **Docs:** `/docs`

Space: `{space_id}`
"""


def main():
    cfg = load_config()
    if cfg.deploy.get("target") != "huggingface_spaces":
        print(f"deploy.target is '{cfg.deploy.get('target')}' — HF deploy skipped.")
        return

    token = os.getenv("HF_TOKEN")
    space_id = os.getenv("SPACE_ID") or cfg.deploy.get("space_id") or ""
    if not token:
        print("HF_TOKEN not set — skipping HF Spaces deploy. "
              "Set HF_TOKEN and deploy.space_id to enable.")
        return
    if not space_id or "/" not in space_id:
        print("deploy.space_id (or SPACE_ID env) must be 'username/space-name' — skipping.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    with open(MODELS_INDEX) as f:
        index = json.load(f)
    manifests = []
    for entry in index["models"]:
        with open(entry["manifest"]) as f:
            manifests.append(json.load(f))
    if not manifests:
        print("No compressed models in index — nothing to deploy.")
        return

    frameworks = sorted({m["framework"] for m in manifests})
    reqs = list(BASE_REQS)
    for fw in frameworks:
        for r in FRAMEWORK_REQS.get(fw, []):
            if r not in reqs:
                reqs.append(r)

    api = HfApi(token=token)
    print(f"Ensuring Space exists: {space_id}")
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker",
                    exist_ok=True, token=token)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def up_bytes(text, path_in_repo):
        api.upload_file(
            path_or_fileobj=io.BytesIO(text.encode("utf-8")),
            path_in_repo=path_in_repo, repo_id=space_id, repo_type="space", token=token,
        )

    # 1. Generated Space files.
    summary = ", ".join(f"{m['name']} ({m['framework']}/{m['task']})" for m in manifests)
    up_bytes(DOCKERFILE, "Dockerfile")
    up_bytes("\n".join(reqs) + "\n", "spaces_requirements.txt")
    up_bytes(readme(cfg.project, "+".join(frameworks), summary, space_id), "README.md")

    # 2. Pipeline code needed to serve.
    for folder in ("serve", "common", "adapters", "compressors"):
        api.upload_folder(
            folder_path=os.path.join(root, folder), path_in_repo=folder,
            repo_id=space_id, repo_type="space", token=token,
            ignore_patterns=["__pycache__/*", "*.pyc"],
        )
    up_bytes(open(os.path.join(root, "pipeline.yaml"), encoding="utf-8").read(), "pipeline.yaml")

    # 3. Only the artifacts the server loads: the index + each model's manifest
    #    and variant files (per-model artifacts/<name>/ dirs).
    needed = {os.path.basename(MODELS_INDEX)}
    for m in manifests:
        for v in [m["native"], *m["variants"]]:
            needed.add(os.path.basename(v["path"]))
        needed.add("variants.json")
    api.upload_folder(
        folder_path=os.path.join(root, "artifacts"), path_in_repo="artifacts",
        repo_id=space_id, repo_type="space", token=token,
        allow_patterns=[f"**/{name}" for name in needed] + list(needed),
    )

    url = f"https://huggingface.co/spaces/{space_id}"
    print(f"\nDeployed. Space building at: {url}")
    print(f"API docs (once built): {url.replace('huggingface.co/spaces', 'huggingface.co/spaces') }/  →  /docs")


if __name__ == "__main__":
    main()
