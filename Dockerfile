# Single shared image for every pipeline stage. Each docker-compose service
# runs the same image with a different command, so the shared common/ adapters/
# compressors/ packages are available everywhere (impossible with per-dir builds).
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; onnxruntime INT8 quantization needs no GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project (see .dockerignore for exclusions).
COPY . .

ENV PIPELINE_CONFIG=/app/pipeline.yaml \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default command trains; compose overrides per stage.
CMD ["python", "train/train.py"]
