#!/usr/bin/env bash
#
# Kompress — one-command local run for Linux (developed/tested on Arch).
#
# Brings up the whole platform: Python venv + deps, web deps, MLflow (best-effort UI),
# the self-serve API in queue mode, a compression worker, and the React dashboard.
#
#   ./run.sh            start everything (prints URLs; Ctrl+C stops all)
#   ./run.sh --with-dl  also install torch + onnxscript (enables the PyTorch path)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
LOGS="$ROOT/.run-logs"
WITH_DL=0
[[ "${1:-}" == "--with-dl" ]] && WITH_DL=1

mkdir -p "$LOGS"

log()  { printf '\033[1;36m[kompress]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[kompress]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[kompress]\033[0m %s\n' "$*" >&2; exit 1; }

# ── prerequisites ────────────────────────────────────────────────────────────
# Prefer a Python the MLflow *server* is happy on (< 3.14); the platform itself
# works on any 3.x. Fall back to whatever python3 exists.
PY="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || true)"
[[ -n "$PY" ]] || die "python3 not found. On Arch:  sudo pacman -S python"
command -v npm >/dev/null || die "npm not found. On Arch:  sudo pacman -S nodejs npm"
log "using $($PY --version) at $PY"

# ── python env ───────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  log "creating venv at .venv"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Only install when something is actually missing — so 2nd+ runs start fast.
if python -c "import mlflow, fastapi, onnxruntime, xgboost, skl2onnx" 2>/dev/null; then
  log "Python dependencies already present — skipping install"
else
  log "installing Python dependencies (first run only; this takes a few minutes)…"
  pip install -q --upgrade pip >/dev/null
  pip install -q -r "$ROOT/requirements.txt"
fi
if [[ "$WITH_DL" == "1" ]] && ! python -c "import torch, onnxscript" 2>/dev/null; then
  log "installing PyTorch (CPU) + onnxscript for the DL path…"
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  pip install -q onnxscript
fi

# Python 3.14 compat: MLflow's server module imports Traversable from importlib.abc,
# which 3.14 removed (it moved to importlib.resources.abc). Idempotent, best-effort.
python - <<'PY' 2>/dev/null || true
import importlib.util, pathlib
spec = importlib.util.find_spec("mlflow")
if spec and spec.submodule_search_locations:
    f = pathlib.Path(list(spec.submodule_search_locations)[0]) / "assistant" / "skill_installer.py"
    if f.exists():
        t = f.read_text()
        if "from importlib.abc import Traversable" in t and "resources.abc" not in t:
            f.write_text(t.replace(
                "from importlib.abc import Traversable",
                "try:\n    from importlib.resources.abc import Traversable\n"
                "except ImportError:\n    from importlib.abc import Traversable"))
            print("[kompress] patched mlflow for Python 3.14")
PY

# ── web deps ─────────────────────────────────────────────────────────────────
if [[ ! -d "$ROOT/web/node_modules" ]]; then
  log "installing web dependencies (npm install)"
  ( cd "$ROOT/web" && npm install --silent )
fi

# ── runtime configuration ────────────────────────────────────────────────────
export PYTHONPATH="$ROOT"
export MLFLOW_EXPERIMENT="self-serve-compression"
export API_RUNS_DIR="$ROOT/api_runs"
export KOMPRESS_QUEUE_DIR="$ROOT/queue"
export KOMPRESS_EXECUTION="queue"
export API_ALLOW_LOCAL_PATHS="1"      # allow the bundled sample models by local path
export CORS_ORIGINS="*"
mkdir -p "$ROOT/mlflow_store" "$ROOT/mlartifacts" "$API_RUNS_DIR" "$KOMPRESS_QUEUE_DIR"

SQLITE_URI="sqlite:///$ROOT/mlflow_store/kompress.db"

PIDS=()
cleanup() {
  log "shutting down…"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── MLflow: try the tracking server (also serves the UI); fall back to sqlite ─
log "starting MLflow server on :5000 (best-effort)…"
mlflow server --backend-store-uri "$SQLITE_URI" \
              --default-artifact-root "$ROOT/mlartifacts" \
              --host 127.0.0.1 --port 5000 >"$LOGS/mlflow.log" 2>&1 &
MLFLOW_PID=$!; PIDS+=("$MLFLOW_PID")

MLFLOW_UP=0
for _ in $(seq 1 15); do
  if curl -sf http://127.0.0.1:5000/health >/dev/null 2>&1; then MLFLOW_UP=1; break; fi
  sleep 1
done

if [[ "$MLFLOW_UP" == "1" ]]; then
  export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
  log "MLflow UI ready at http://localhost:5000"
else
  warn "MLflow server did not start (Python 3.14 is a known cause; see .run-logs/mlflow.log)."
  warn "Falling back to a direct sqlite store — the platform still works, just no MLflow UI."
  kill "$MLFLOW_PID" 2>/dev/null || true
  export MLFLOW_TRACKING_URI="$SQLITE_URI"
fi

# ── API (queue mode) + worker + dashboard ────────────────────────────────────
log "starting API on :8000"
uvicorn api.app:app --host 127.0.0.1 --port 8000 --log-level warning >"$LOGS/api.log" 2>&1 &
PIDS+=("$!")

log "starting compression worker"
python -m worker.worker --poll 2 >"$LOGS/worker.log" 2>&1 &
PIDS+=("$!")

log "starting dashboard on :5173"
( cd "$ROOT/web" && npm run dev -- --port 5173 --host 127.0.0.1 ) >"$LOGS/web.log" 2>&1 &
PIDS+=("$!")

sleep 4
cat <<EOF

  ┌───────────────────────────────────────────────────────────────┐
  │  Kompress is running                                          │
  │                                                               │
  │    Dashboard   http://localhost:5173                          │
  │    API + docs  http://localhost:8000/docs                     │
  │    MLflow UI   http://localhost:5000  $( [[ $MLFLOW_UP == 1 ]] && echo '(up)          ' || echo '(unavailable) ')│
  │                                                               │
  │  Test it: open the Dashboard → Submit →                       │
  │    model  artifacts/churn/model.pkl                           │
  │    test   data/test.csv                                       │
  │    xgboost / binary_classification / target: churn / cpu-generic
  │                                                               │
  │  Logs in .run-logs/   ·   Ctrl+C to stop everything           │
  └───────────────────────────────────────────────────────────────┘

EOF

wait
