"""Feature drift detection using Population Stability Index (PSI).

Compares incoming prediction-time features against train distribution.
Runs as a scheduled job or triggered post-deploy.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config, feature_names  # noqa: E402

PSI_WARN  = float(os.getenv("PSI_WARN", "0.1"))   # moderate drift
PSI_ALERT = float(os.getenv("PSI_ALERT", "0.2"))  # severe drift
N_BINS    = int(os.getenv("N_BINS", "10"))


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI = sum((actual% - expected%) * ln(actual% / expected%))"""
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints  = np.unique(breakpoints)  # remove duplicates for low-cardinality features

    exp_counts  = np.histogram(expected, bins=breakpoints)[0]
    act_counts  = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Clip to avoid log(0)
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def reconstruct_train_sample(stats: dict, cols: list, n: int = 5000) -> pd.DataFrame:
    """Approximate train distribution from saved describe() stats via normal sampling."""
    rng = np.random.default_rng(0)
    rows = {}
    for col in cols:
        if col in stats:
            rows[col] = rng.normal(stats[col].get("mean", 0), stats[col].get("std", 1) or 1, n)
    return pd.DataFrame(rows)


def drift_for_model(spec) -> dict | None:
    stats_path = os.path.join(spec.out_dir, "train_stats.json")
    incoming_csv = os.getenv("INCOMING_CSV", spec.test_csv)
    if not os.path.exists(stats_path):
        print(f"[{spec.name}] no train_stats.json (BYO artifact?) — skipping drift.")
        return None

    cols = feature_names(spec.features)
    with open(stats_path) as f:
        stats = json.load(f)
    incoming = pd.read_csv(incoming_csv)
    train_df = reconstruct_train_sample(stats, cols)

    report = {"model": spec.name, "features": {}, "summary": {}}
    psi_vals = []
    for col in cols:
        if col not in incoming.columns or col not in train_df.columns:
            continue
        psi = compute_psi(train_df[col].values, incoming[col].values, n_bins=N_BINS)
        psi_vals.append(psi)
        status = "alert" if psi >= PSI_ALERT else ("warn" if psi >= PSI_WARN else "ok")
        report["features"][col] = {"psi": round(psi, 4), "status": status}

    alert_cols = [c for c, v in report["features"].items() if v["status"] == "alert"]
    warn_cols = [c for c, v in report["features"].items() if v["status"] == "warn"]
    report["summary"] = {
        "mean_psi": round(float(np.mean(psi_vals)) if psi_vals else 0.0, 4),
        "alert_features": alert_cols, "warn_features": warn_cols,
        "overall_status": "alert" if alert_cols else ("warn" if warn_cols else "ok"),
        "n_incoming": len(incoming),
    }

    with open(os.path.join(spec.out_dir, "drift_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{spec.name}] drift status: {report['summary']['overall_status'].upper()} "
          f"(mean PSI={report['summary']['mean_psi']})")
    return report


def main():
    cfg = load_config()
    for spec in cfg.models:
        drift_for_model(spec)


if __name__ == "__main__":
    main()
