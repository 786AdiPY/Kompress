"""Kompress compression worker: claim a job from the queue, run the compression
engine, store the result in MLflow (delta metrics + registered ONNX + status tag).

Run one or many — each job is claimed by exactly one worker. In production, run each
worker as an isolated/ephemeral container (ideally a Kubernetes Job per message) so
an untrusted uploaded model is sandboxed away from the API and other jobs.

    python -m worker.worker            # loop forever, polling the queue
    python -m worker.worker --drain    # process everything pending, then exit
    python -m worker.worker --once     # process exactly one job, then exit
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from worker.executor import execute_job     # noqa: E402
from worker.queue import get_queue           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poll", type=float, default=2.0, help="Seconds between polls when idle.")
    ap.add_argument("--drain", action="store_true", help="Exit once the queue is empty.")
    ap.add_argument("--once", action="store_true", help="Process a single job, then exit.")
    args = ap.parse_args()

    q = get_queue()
    print(f"[worker] started; queue={q}", flush=True)

    processed = 0
    while True:
        claimed = q.claim()
        if claimed is None:
            if args.drain:
                print(f"[worker] queue empty; drained {processed} job(s). Exiting.", flush=True)
                return
            time.sleep(args.poll)
            continue

        payload, handle = claimed
        run_id = payload.get("run_id", "?")
        model = payload.get("model_name", "?")
        print(f"[worker] claimed run={run_id} model={model}", flush=True)
        try:
            status = execute_job(run_id, payload["job"], payload["model_name"])
            q.complete(handle)
            print(f"[worker] done run={run_id} -> {status}", flush=True)
        except Exception as e:  # noqa: BLE001 — execute_job already tags failure; guard the loop
            q.fail(handle)
            print(f"[worker] ERROR run={run_id}: {e}", flush=True)

        processed += 1
        if args.once:
            return


if __name__ == "__main__":
    main()
