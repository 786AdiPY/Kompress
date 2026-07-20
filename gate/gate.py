"""Quality gate for every model. Blocks deploy if any compressed variant
degrades too much vs its native baseline. Task-aware (classification: acc/AUC
drop; regression: RMSE rise). Writes per-model artifacts/<name>/gate_report.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config  # noqa: E402


def find_baseline(results):
    for r in results:
        if r.get("kind") == "native" or r["model"].startswith("native_"):
            return r
    return results[0] if results else None


def gate_model(spec, results) -> dict:
    baseline = find_baseline(results)
    if baseline is None:
        return {"model": spec.name, "passed": False, "reason": "no native baseline"}

    task = spec.task
    max_acc_drop = float(spec.gate.get("max_acc_drop", 0.01))
    max_auc_drop = float(spec.gate.get("max_auc_drop", 0.01))
    max_rmse_rise = float(spec.gate.get("max_rmse_rise", 0.05))

    report = {"model": spec.name, "baseline": baseline["model"], "task": task,
              "checks": [], "passed": True}

    for r in results:
        if r["model"] == baseline["model"]:
            continue
        check = {"variant": r["model"], "latency_ms": r["latency_ms"],
                 "speedup": r.get("speedup_vs_native", 1.0), "passed": True, "fail_reason": []}

        if task == "regression":
            rise = r.get("rmse", 0.0) - baseline.get("rmse", 0.0)
            check["rmse_rise"] = round(rise, 4)
            if rise > max_rmse_rise:
                check["passed"] = False
                check["fail_reason"].append(f"RMSE rise {rise:.4f} > {max_rmse_rise}")
        else:
            acc_drop = baseline.get("accuracy", 0.0) - r.get("accuracy", 0.0)
            auc_drop = baseline.get("auc", 0.0) - r.get("auc", 0.0)
            check.update({"acc_drop": round(acc_drop, 4), "auc_drop": round(auc_drop, 4)})
            if acc_drop > max_acc_drop:
                check["passed"] = False
                check["fail_reason"].append(f"accuracy drop {acc_drop:.4f} > {max_acc_drop}")
            if auc_drop > max_auc_drop:
                check["passed"] = False
                check["fail_reason"].append(f"AUC drop {auc_drop:.4f} > {max_auc_drop}")

        if not check["fail_reason"]:
            check.pop("fail_reason")
        else:
            report["passed"] = False
        report["checks"].append(check)

    return report


def main():
    cfg = load_config()
    all_passed = True
    for spec in cfg.models:
        results_path = os.path.join(spec.out_dir, "benchmark_results.json")
        if not os.path.exists(results_path):
            print(f"[{spec.name}] no benchmark results — skipping gate.")
            continue
        with open(results_path) as f:
            results = json.load(f)

        report = gate_model(spec, results)
        with open(os.path.join(spec.out_dir, "gate_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        status = "PASSED" if report["passed"] else "FAILED"
        print(f"[{spec.name}] GATE {status}")
        for c in report.get("checks", []):
            flag = "ok" if c["passed"] else "FAIL " + "; ".join(c.get("fail_reason", []))
            print(f"    {c['variant']:<26} {flag}")
        all_passed = all_passed and report["passed"]

    if all_passed:
        print("\nALL GATES PASSED.")
        sys.exit(0)
    print("\nONE OR MORE GATES FAILED.")
    sys.exit(1)


if __name__ == "__main__":
    main()
