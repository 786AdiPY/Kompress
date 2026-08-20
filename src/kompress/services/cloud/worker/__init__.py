"""Kompress compression worker layer.

Decouples job submission (the API enqueues) from job execution (a worker runs the
compression engine in isolation and stores results in MLflow). The API stays a thin
enqueuer; workers scale independently and can run as isolated/ephemeral containers
(ideally one Kubernetes Job per compression) so an untrusted uploaded model can't
affect the API or other jobs.

  worker/executor.py  — the shared "run one compression job" logic (used by the
                        worker, and by the API's inline fallback mode)
  worker/queue.py     — pluggable job queue; default is a filesystem queue (no deps)
  worker/worker.py    — the worker loop: claim a job -> execute -> store in MLflow
"""
