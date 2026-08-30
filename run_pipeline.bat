@echo off
setlocal enabledelayedexpansion

set ROOT=%~dp0
cd /d "%ROOT%"
set PYTHONUTF8=1
set MLFLOW_TRACKING_URI=http://localhost:5000
set DATA_DIR=data
set MODEL_OUT=artifacts/model.pkl
set MODEL_PKL=artifacts/model.pkl
set MODEL_ONNX=artifacts/model_fp32.onnx
set MODEL_TRT=artifacts/model_int8.trt
set TEST_CSV=data/test.csv
set RESULTS_OUT=artifacts/benchmark_results.json
set RESULTS_PATH=artifacts/benchmark_results.json
set GATE_REPORT_OUT=artifacts/gate_report.json
set GATE_REPORT=artifacts/gate_report.json
set META_PATH=artifacts/model_meta.json
set TRAIN_STATS_PATH=artifacts/model_train_stats.json
set DRIFT_REPORT_OUT=artifacts/drift_report.json

echo ============================================================
echo  Model Compression MLOps Pipeline (multi-framework)
echo ============================================================

:: Step 0 — create dirs + ensure deps
mkdir artifacts 2>nul
mkdir mlflow_store\artifacts 2>nul
mkdir data 2>nul

echo [0/9] Checking Python dependencies...
python -c "import mlflow, xgboost, onnxruntime, skl2onnx, fastapi, yaml" 2>nul
if errorlevel 1 (
    echo Installing requirements ^(first run^)...
    python -m pip install -r requirements.txt
    if errorlevel 1 ( echo FAILED: pip install & goto :error )
)

:: Step 1 — Start MLflow in background
echo [1/9] Starting MLflow server...
start "MLflow" cmd /k "cd /d "%ROOT%" && set PYTHONUTF8=1 && python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow_store/mlflow.db --default-artifact-root ./mlflow_store/artifacts"
echo Waiting for MLflow to start...
timeout /t 10 /nobreak >nul

set PYTHONPATH=%ROOT%src;%ROOT%

:: Step 2 — Generate data
echo [2/9] Generating synthetic dataset...
python data/generate.py --n 10000 --out data
if errorlevel 1 ( echo FAILED: data generation & goto :error )
echo Done.

:: Step 3 — Train (classic-ML baseline; skip if you bring your own model)
echo [3/9] Training baseline model...
python -m kompress.tools.train
if errorlevel 1 ( echo FAILED: training & goto :error )
echo Done.

:: Step 4 — Compress: export ONNX FP32 + run every configured compressor
echo [4/9] Compressing (ONNX FP32 + INT8 quantization + TRT if GPU)...
python -m kompress.engine.pipeline.compress
if errorlevel 1 ( echo FAILED: compression & goto :error )
echo Done.

:: Step 5 — Benchmark all variants
echo [5/9] Benchmarking native + all compressed variants...
python -m kompress.engine.pipeline.benchmark
if errorlevel 1 ( echo FAILED: benchmark & goto :error )
echo Done.

:: Step 6 — Quality gate
echo [6/9] Running quality gate...
python -m kompress.engine.pipeline.gate
if errorlevel 1 ( echo FAILED: quality gate blocked deployment & goto :error )
echo Done.

:: Step 7 — Register best variant
echo [7/9] Registering best variant to MLflow...
python -m kompress.engine.registry.register
if errorlevel 1 ( echo FAILED: registry & goto :error )
echo Done.

:: Step 8 — Deploy to Hugging Face Spaces (skipped unless HF_TOKEN + space_id set)
echo [8/9] Deploying to Hugging Face Spaces...
python -m kompress.tools.hf_deploy
echo Done.

:: Step 9 — Drift monitor
echo [9/9] Running drift detection...
set INCOMING_CSV=data/test.csv
python -m kompress.tools.monitor
echo Done.

echo.
echo ============================================================
echo  Pipeline complete. Starting API server...
echo  Swagger UI  : http://localhost:8000/docs
echo  MLflow UI   : http://localhost:5000
echo ============================================================
echo.

:: Start serve in new window
start "Serve" cmd /k "cd /d "%ROOT%" && set PYTHONUTF8=1 && set MODELS_INDEX=artifacts/models.json && uvicorn kompress.services.cloud.serve.app:app --host 0.0.0.0 --port 8000"

echo API server starting in new window...
echo Press any key to exit this window (server keeps running).
pause >nul
exit /b 0

:error
echo.
echo Pipeline stopped due to error. Check output above.
pause
exit /b 1
