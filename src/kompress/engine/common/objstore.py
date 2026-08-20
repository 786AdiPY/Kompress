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
    endpoint = os.getenv("S3_ENDPOINT_URL")  # e.g. http://minio:9000 for MinIO
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
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


def download(uri: str, dest_dir: str) -> str:
    """Download an s3:// object to dest_dir, returning the local path."""
    os.makedirs(dest_dir, exist_ok=True)
    bucket, key = parse_s3_uri(uri)
    local = os.path.join(dest_dir, os.path.basename(key) or "object")
    s3_client().download_file(bucket, key, local)
    return local


def upload_fileobj(fileobj, key: str, bucket: str = DEFAULT_BUCKET) -> str:
    """Stream an open file object into object storage; returns its s3:// pointer.
    This is what turns a UI file upload into a pointer the job can consume."""
    ensure_bucket(bucket)
    s3_client().upload_fileobj(fileobj, bucket, key)
    return f"s3://{bucket}/{key}"
