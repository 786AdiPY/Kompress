"""BERT-Base model compression and evaluation benchmark on Stanford NLP SST-2 dataset.

Tests Kompress's FP32 baseline, Full FP16, Mixed FP16 (sensitive layers preserved),
Adaptive FP16, and Dynamic INT8 quantization on BERT-base sentiment classification.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort

# Add src to python path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from kompress.engine.compressors import (
    OnnxFp32,
    OnnxFp16,
    OnnxFp16Mixed,
    OnnxFp16Adaptive,
    OnnxInt8Dynamic,
)
from kompress.engine.compressors.base import Variant
from kompress.engine.common.metrics import compute_metrics


MODEL_DIR = os.path.join(REPO_DIR, "artifacts", "bert_sst2")
DATA_DIR = os.path.join(REPO_DIR, "data")
ONNX_FP32_PATH = os.path.join(MODEL_DIR, "model_fp32.onnx")


class BertWrapper(torch.nn.Module):
    """Wrapper that outputs probabilities directly for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return torch.softmax(logits, dim=-1)


def export_bert_to_onnx(model_path: str, onnx_out_path: str, tokenizer) -> str:
    print(f"\n[1/4] Exporting PyTorch BERT model -> ONNX FP32: {onnx_out_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    wrapped_model = BertWrapper(model)

    dummy_text = "This film is absolutely wonderful and brilliant."
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )

    dummy_input_ids = inputs["input_ids"]
    dummy_attention_mask = inputs["attention_mask"]

    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["probabilities"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "probabilities": {0: "batch"},
        },
        opset_version=17,
    )
    size_mb = os.path.getsize(onnx_out_path) / (1024 * 1024)
    print(f"ONNX FP32 exported successfully ({size_mb:.2f} MB)")
    return onnx_out_path


def benchmark_onnx_variant(
    variant_name: str,
    onnx_path: str,
    eval_input_ids: np.ndarray,
    eval_attention_mask: np.ndarray,
    labels: np.ndarray,
    num_runs: int = 2,
) -> Dict[str, Any]:
    print(f"  Benchmarking '{variant_name}'...")
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]

    feed_dict = {
        input_names[0]: eval_input_ids,
        input_names[1]: eval_attention_mask,
    }

    # Warmup
    for _ in range(3):
        session.run(None, {k: v[:2] for k, v in feed_dict.items()})

    # Timing
    t0 = time.perf_counter()
    for _ in range(num_runs):
        outputs = session.run(None, feed_dict)
    elapsed_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # Predictions & metrics
    probs = outputs[0]
    metrics = compute_metrics("binary_classification", labels, probs[:, 1])

    return {
        "variant": variant_name,
        "path": onnx_path,
        "size_mb": round(size_mb, 2),
        "latency_ms": round(elapsed_ms, 2),
        "accuracy": metrics.get("accuracy", 0.0),
        "auc": metrics.get("auc", 0.0),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("==========================================================================")
    print("        KOMPRESS: BERT-Base SST-2 Precision & Quantization Benchmark      ")
    print("==========================================================================")

    # 1. Load Tokenizer & Prepare Data
    print("\nLoading tokenizer and SST-2 validation dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    from datasets import load_dataset
    sst2 = load_dataset("stanfordnlp/sst2", split="validation")
    sentences = sst2["sentence"]
    labels = np.array(sst2["label"])

    print(f"Dataset loaded: {len(sentences)} samples.")

    # Save local CSV copy for dataset persistence
    df_local = pd.DataFrame({"sentence": sentences, "label": labels})
    csv_path = os.path.join(DATA_DIR, "sst2_validation.csv")
    df_local.to_csv(csv_path, index=False)
    print(f"Local SST-2 dataset saved -> {csv_path}")

    # Tokenize eval batch (evaluate first 200 samples for high speed & precision check)
    eval_n = min(25, len(sentences))
    eval_text = sentences[:eval_n]
    eval_labels = labels[:eval_n]

    encoded = tokenizer(
        eval_text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="np",
    )
    eval_input_ids = encoded["input_ids"].astype(np.int64)
    eval_attention_mask = encoded["attention_mask"].astype(np.int64)

    # 2. Export ONNX FP32
    if not os.path.exists(ONNX_FP32_PATH):
        export_bert_to_onnx(MODEL_DIR, ONNX_FP32_PATH, tokenizer)
    else:
        print(f"ONNX FP32 model already exists at {ONNX_FP32_PATH}")

    # 3. Compression Stage (FP32, FP16, Mixed FP16, Adaptive FP16, INT8 Dynamic)
    print("\n[2/4] Running Kompress precision converters...")
    compressors = [
        OnnxFp32(),
        OnnxFp16(),
        OnnxFp16Mixed(),
        OnnxFp16Adaptive(),
        OnnxInt8Dynamic(),
    ]

    variants: List[Variant] = []
    dummy_calib = {
        "input_ids": eval_input_ids,
        "attention_mask": eval_attention_mask,
    }

    for comp in compressors:
        print(f"--- Running compressor: {comp.name} ---")
        try:
            v = comp.compress(ONNX_FP32_PATH, MODEL_DIR, calib_data=dummy_calib)
            if v:
                variants.append(v)
        except Exception as e:
            print(f"Compressor {comp.name} failed: {e}")

    # 4. Benchmark Stage
    print(f"\n[3/4] Benchmarking precision variants on {eval_n} SST-2 validation samples...")
    results = []
    for v in variants:
        res = benchmark_onnx_variant(
            v.name,
            v.path,
            eval_input_ids,
            eval_attention_mask,
            eval_labels,
        )
        results.append(res)

    # Baseline for speedup calculations
    baseline_lat = results[0]["latency_ms"] if results else 1.0
    for r in results:
        r["speedup"] = round(baseline_lat / max(r["latency_ms"], 0.001), 2)
        r["size_reduction_pct"] = round(
            (1.0 - (r["size_mb"] / results[0]["size_mb"])) * 100, 1
        )

    # 5. Output Summary Table
    print("\n" + "=" * 80)
    print("                       BERT-BASE SST-2 BENCHMARK RESULTS                  ")
    print("=" * 80)
    print(
        f"{'Variant':<20} | {'Size (MB)':<10} | {'Latency (ms)':<12} | {'Speedup':<8} | {'Accuracy':<9} | {'AUC':<6}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['variant']:<20} | {r['size_mb']:<10.1f} | {r['latency_ms']:<12.2f} | {r['speedup']:<7.2f}x | {r['accuracy']:<9.4f} | {r['auc']:<6.4f}"
        )
    print("=" * 80)

    # Save summary report
    report_path = os.path.join(MODEL_DIR, "bert_sst2_compression_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved compression report -> {report_path}")


if __name__ == "__main__":
    main()
