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


def default_queue_dir() -> str:
    return os.getenv("KOMPRESS_QUEUE_DIR", os.path.join(REPO_ROOT, "queue"))


def get_queue() -> QueueBackend:
    """Resolve the configured queue backend. KOMPRESS_QUEUE selects it:
    'file' (default) -> FileQueue(KOMPRESS_QUEUE_DIR); 'redis://...' -> not yet
    implemented (add RedisQueue here for production)."""
    backend = os.getenv("KOMPRESS_QUEUE", "file")
    if backend in ("file", "filesystem") or backend.startswith("file:"):
        return FileQueue(default_queue_dir())
    if backend.startswith("redis"):
        raise NotImplementedError(
            "Redis/RQ backend not implemented — the filesystem queue is the default. "
            "Add a RedisQueue(QueueBackend) here to enable it.")
    raise ValueError(f"Unknown KOMPRESS_QUEUE backend '{backend}'.")
