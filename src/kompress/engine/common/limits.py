"""Size-tier limits shared by the API and deployment config.

Kompress's primary path is DIRECT UPLOAD -> object storage for models under
MAX_UPLOAD_BYTES (default 10GB): the user hands us a file, we store it, we process
it from there. Models at or above that size are pointer-only (the caller hosts the
file themselves and gives us an s3://... ref) — supported by POST /runs today, just
not the tuned/tested path right now.

Worker resource sizing (see deploy/docker-compose.oss.yml) follows a rule of thumb
for the upload tier: ~3-4x model size in scratch DISK (original + ONNX FP32 export +
2-3 compressed variants + calibration data) and ~2-3x model size in MEMORY (loading
+ exporting the model, framework-dependent). At the 10GB ceiling that's ~30-40GB
disk and ~20-30GB RAM per concurrent job.
"""
import os

GB = 1024 ** 3

MAX_UPLOAD_BYTES = int(float(os.getenv("KOMPRESS_MAX_UPLOAD_GB", "10")) * GB)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
