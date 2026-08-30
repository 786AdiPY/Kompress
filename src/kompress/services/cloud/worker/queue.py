"""Pluggable job queue between the API (enqueue) and the worker (consume).

Default backend is a FILESYSTEM queue — no Redis, no broker, no external service —
so it runs anywhere out of the box and is trivially inspectable (jobs are just JSON
files moving between pending/ -> processing/ -> done|failed/). A job is claimed by
an atomic rename, so exactly one worker gets it even with several running.

Production backends (Redis/RQ, a cloud queue, or one Kubernetes Job per message)
implement the same QueueBackend interface and swap in via KOMPRESS_QUEUE — nothing
else changes.
"""
from __future__ import annotations

import glob
import json
import os
import time
import uuid
from abc import ABC, abstractmethod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class QueueBackend(ABC):
    @abstractmethod
    def enqueue(self, payload: dict) -> str: ...
    @abstractmethod
    def claim(self): ...          # -> (payload: dict, handle) | None
    @abstractmethod
    def complete(self, handle): ...
    @abstractmethod
    def fail(self, handle): ...
    def has_pending(self) -> bool:
        return False


class FileQueue(QueueBackend):
    """A directory-backed FIFO queue. Ordering is by enqueue time (ms) embedded in
    the filename; claiming is an atomic os.rename into processing/."""

    def __init__(self, root: str):
        self.root = root
        self.pending = os.path.join(root, "pending")
        self.processing = os.path.join(root, "processing")
        self.done = os.path.join(root, "done")
        self.failed = os.path.join(root, "failed")
        for d in (self.pending, self.processing, self.done, self.failed):
            os.makedirs(d, exist_ok=True)

    def __repr__(self):
        return f"FileQueue({self.root})"

    def enqueue(self, payload: dict) -> str:
        jid = payload.get("run_id") or uuid.uuid4().hex
        name = f"{int(time.time() * 1000)}_{jid}.json"
        tmp = os.path.join(self.pending, "." + name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.rename(tmp, os.path.join(self.pending, name))  # atomic publish
        return jid

    def claim(self):
        files = sorted(glob.glob(os.path.join(self.pending, "*.json")),
                       key=lambda p: os.path.basename(p))  # ms-prefixed => FIFO
        for src in files:
            dst = os.path.join(self.processing, os.path.basename(src))
            try:
                os.rename(src, dst)  # atomic claim; loses the race -> try next
            except OSError:
                continue
            with open(dst) as f:
                return json.load(f), dst
        return None

    def complete(self, handle):
        os.rename(handle, os.path.join(self.done, os.path.basename(handle)))

    def fail(self, handle):
        os.rename(handle, os.path.join(self.failed, os.path.basename(handle)))

    def has_pending(self) -> bool:
        return bool(glob.glob(os.path.join(self.pending, "*.json")))


class RedisQueue(QueueBackend):
    """Reliable Redis/Valkey-backed queue for multi-node deployments.

    Uses a reliable-queue pattern: LMOVE atomically claims from the pending list into
    a processing list, so a crashed worker's job is not lost (it can be requeued from
    'processing'). Works against Redis or its OSS fork Valkey — same protocol.
    """

    def __init__(self, url: str, namespace: str = "kompress"):
        import redis  # lazy: only needed when this backend is selected
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.pending = f"{namespace}:pending"
        self.processing = f"{namespace}:processing"
        self.failed = f"{namespace}:failed"

    def __repr__(self):
        return f"RedisQueue({self.pending})"

    def enqueue(self, payload: dict) -> str:
        raw = json.dumps(payload)
        self.r.lpush(self.pending, raw)
        return payload.get("run_id") or uuid.uuid4().hex

    def claim(self):
        # atomic pending -> processing; handle is the raw string (needed to ack).
        raw = self.r.lmove(self.pending, self.processing, "RIGHT", "LEFT")
        if raw is None:
            return None
        return json.loads(raw), raw

    def complete(self, handle):
        self.r.lrem(self.processing, 1, handle)

    def fail(self, handle):
        self.r.lrem(self.processing, 1, handle)
        self.r.lpush(self.failed, handle)

    def has_pending(self) -> bool:
        return self.r.llen(self.pending) > 0


def default_queue_dir() -> str:
    return os.getenv("KOMPRESS_QUEUE_DIR", os.path.join(REPO_ROOT, "queue"))


def get_queue() -> QueueBackend:
    """Resolve the configured queue backend from KOMPRESS_QUEUE:
      'file' (default)  -> FileQueue(KOMPRESS_QUEUE_DIR)   — no external service
      'redis://host:port/0' | 'valkey://...' -> RedisQueue — multi-node production
    """
    backend = os.getenv("KOMPRESS_QUEUE", "file")
    if backend in ("file", "filesystem") or backend.startswith("file:"):
        return FileQueue(default_queue_dir())
    if backend.startswith("redis") or backend.startswith("valkey"):
        url = backend.replace("valkey://", "redis://", 1)
        return RedisQueue(url)
    raise ValueError(f"Unknown KOMPRESS_QUEUE backend '{backend}'.")
