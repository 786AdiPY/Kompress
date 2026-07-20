"""Built-in classic-ML baseline trainer (XGBoost).

Trains every model in pipeline.yaml marked `trainable: true` whose framework is
xgboost, using that model's own features/target/data, and writes its artifact +
drift baseline stats into the model's artifacts/<name>/ dir. Models of other
frameworks (e.g. pytorch) are trained externally and dropped in as artifacts.
"""
import os
import sys
import json
import pickle

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config, feature_names  # noqa: E402

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

PARAMS = {
    "n_estimators":     int(os.getenv("N_ESTIMATORS", "300")),
    "max_depth":        int(os.getenv("MAX_DEPTH", "6")),
    "learning_rate":  float(os.getenv("LEARNING_RATE", "0.05")),
    "subsample":      float(os.getenv("SUBSAMPLE", "0.8")),
    "colsample_bytree": float(os.getenv("COLSAMPLE", "0.8")),
    "eval_metric":      "logloss",
    "tree_method":      "hist",
    "random_state":     42,
}


def train_xgb(spec, experiment):
    cols, target = feature_names(spec.features), spec.target
    train = pd.read_csv(spec.train_csv)
    test = pd.read_csv(spec.test_csv)
    X_tr, y_tr = train[cols].values, train[target].values
    X_te, y_te = test[cols].values, test[target].values
    os.makedirs(spec.out_dir, exist_ok=True)

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"train-{spec.name}") as run:
        mlflow.log_params(PARAMS)
        mlflow.set_tag("model", spec.name)

        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        mlflow.log_metrics({"test_accuracy": acc, "test_auc": auc, "test_f1": f1})
        print(f"[{spec.name}] ACC={acc:.4f} AUC={auc:.4f} F1={f1:.4f}")

        os.makedirs(os.path.dirname(spec.artifact) or ".", exist_ok=True)
        with open(spec.artifact, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(spec.artifact, artifact_path="model")

        # Drift baseline stats for the monitor.
        stats = pd.DataFrame(X_tr, columns=cols).describe().to_dict()
        stats_path = os.path.join(spec.out_dir, "train_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[{spec.name}] artifact -> {spec.artifact}")


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = os.getenv("MLFLOW_EXPERIMENT", f"{cfg.project}-compression")

    trained = 0
    for spec in cfg.models:
        if not spec.trainable:
            print(f"[{spec.name}] not trainable (bring your own artifact) — skipping.")
            continue
        if spec.framework != "xgboost":
            print(f"[{spec.name}] built-in trainer only supports xgboost "
                  f"(framework={spec.framework}) — provide a trained artifact.")
            continue
        train_xgb(spec, experiment)
        trained += 1

    print(f"\nTrained {trained} model(s).")


if __name__ == "__main__":
    main()
