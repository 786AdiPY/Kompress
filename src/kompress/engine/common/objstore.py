"""S3-compatible object storage connector (AWS S3 or self-hosted MinIO / Ceph).

One client, endpoint-aware: set S3_ENDPOINT_URL to point at MinIO; leave it unset for
real AWS S3. Used for two things:
  * resolving submitted `s3://` pointers to a local file (download), and
  * the UI upload path — a user's uploaded .pkl is streamed here and becomes an
    `s3://` pointer, so the engine still only ever sees pointers, never raw bytes.

Credentials come from the standard AWS_* env vars (or an instance/pod role).
"""
from __future__ import annotations

import os

DEFAULT_BUCKET = os.getenv("KOMPRESS_BUCKET", "kompress")


def s3_client():
    import boto3
    from botocore.config import Config
    endpoint = os.getenv("S3_ENDPOINT_URL")
    # Strict 5s connection / 10s read timeout & 1 attempt so slow/unreachable S3 never hangs the UI
    config = Config(
        connect_timeout=5,
        read_timeout=10,
        retries={"max_attempts": 1},
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        config=config,
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """s3://bucket/key -> (bucket, key)."""
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def ensure_bucket(bucket: str = DEFAULT_BUCKET) -> None:
    c = s3_client()
    try:
        c.head_bucket(Bucket=bucket)
    except Exception:
        c.create_bucket(Bucket=bucket)


def get_repo_root():
    cur = os.path.abspath(__file__)
    while os.path.dirname(cur) != cur:
        if os.path.exists(os.path.join(cur, "pyproject.toml")) or os.path.exists(os.path.join(cur, "pipeline.yaml")):
            return cur
        cur = os.path.dirname(cur)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

REPO_ROOT = get_repo_root()


def is_s3_configured() -> bool:
    """Check if MinIO/S3 object storage is explicitly configured and enabled."""
    endpoint = os.getenv("S3_ENDPOINT_URL")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    return bool(endpoint or access_key)


def download(uri: str, dest_dir: str) -> str:
    """Download an s3:// object to dest_dir, returning the local path."""
    os.makedirs(dest_dir, exist_ok=True)
    bucket, key = parse_s3_uri(uri)
    local = os.path.join(dest_dir, os.path.basename(key) or "object")
    
    local_fallback = os.path.join(REPO_ROOT, "artifacts", key)
    if os.path.exists(local_fallback):
        import shutil
        shutil.copyfile(local_fallback, local)
        return local
        
    if is_s3_configured():
        try:
            s3_client().download_file(bucket, key, local)
            return local
        except Exception:
            pass

    if os.path.exists(local_fallback):
        import shutil
        shutil.copyfile(local_fallback, local)
        return local
    raise FileNotFoundError(f"Object {uri} not found in S3 or local storage fallback.")


def upload_fileobj(fileobj, key: str, bucket: str = DEFAULT_BUCKET) -> str:
    """Stream an open file object into object storage; returns its pointer.
    If S3/MinIO is unreachable or unconfigured, streams directly to local disk in artifacts/ uploads."""
    if is_s3_configured():
        try:
            ensure_bucket(bucket)
            s3_client().upload_fileobj(fileobj, bucket, key)
            return f"s3://{bucket}/{key}"
        except Exception as e:
            print(f"[objstore] S3 upload failed ({e}); falling back to local storage.", flush=True)

    # Reset file pointer if partially read
    if hasattr(fileobj, "seek"):
        try:
            fileobj.seek(0)
        except Exception:
            pass
    if hasattr(fileobj, "_f") and hasattr(fileobj._f, "seek"):
        try:
            fileobj._f.seek(0)
        except Exception:
            pass

    local_dir = os.path.join(REPO_ROOT, "artifacts", os.path.dirname(key))
    os.makedirs(local_dir, exist_ok=True)
    local_file = os.path.join(REPO_ROOT, "artifacts", key)
    with open(local_file, "wb") as f:
        if hasattr(fileobj, "read"):
            while True:
                chunk = fileobj.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        else:
            f.write(fileobj)
    return f"artifacts/{key}"
