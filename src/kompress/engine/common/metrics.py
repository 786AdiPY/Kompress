"""Task-aware metric computation shared by benchmark and gate."""
from __future__ import annotations

import numpy as np


def compute_metrics(task: str, y_true, y_pred_proba) -> dict:
    """y_pred_proba: 1-D positive-class prob (binary), (n,C) matrix (multiclass),
    or 1-D predictions (regression)."""
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(y_pred_proba)

    if task == "regression":
        if p.ndim > 1:
            if p.shape[1] == 1:
                p = p.ravel()
            else:
                p = p.argmax(axis=1) if p.dtype.kind not in 'fc' else p[:, 0]
        else:
            p = p.ravel()
        if len(p) != len(y_true):
            min_len = min(len(p), len(y_true))
            p, y_true = p[:min_len], y_true[:min_len]
        err = p.astype(float) - y_true.astype(float)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}

    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    if task == "binary_classification":
        if p.ndim > 1 and p.shape[1] == 2:
            p = p[:, 1]
        elif p.ndim > 1 and p.shape[1] == 1:
            p = p.ravel()
        if len(p) != len(y_true):
            min_len = min(len(p), len(y_true))
            p, y_true = p[:min_len], y_true[:min_len]
        pred = (p > 0.5).astype(int) if (p.min() >= 0 and p.max() <= 1) else (p > 0).astype(int)
        out = {"accuracy": round(float(accuracy_score(y_true, pred)), 4),
               "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4)}
        try:
            out["auc"] = round(float(roc_auc_score(y_true, p)), 4)
        except Exception:
            out["auc"] = 0.0
        return out

    # multiclass
    if p.ndim > 1 and p.shape[1] > 1:
        pred = p.argmax(axis=1)
    else:
        pred = (p.ravel() > 0.5).astype(int)
    if len(pred) != len(y_true):
        min_len = min(len(pred), len(y_true))
        pred, y_true = pred[:min_len], y_true[:min_len]
    out = {"accuracy": round(float(accuracy_score(y_true, pred)), 4),
           "f1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4)}
    try:
        out["auc"] = round(float(roc_auc_score(y_true, p, multi_class="ovr")), 4)
    except Exception:
        out["auc"] = 0.0
    return out


def higher_is_better(metric: str) -> bool:
    return metric not in ("rmse", "mae")
