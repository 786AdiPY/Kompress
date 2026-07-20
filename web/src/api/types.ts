// Shared API contract types. This file is OWNED by the foundation.
// Page agents import from here; they must NOT edit it.

export type RunStatus =
  | 'pending_gate'
  | 'pending_approval'
  | 'approved'
  | 'rejected'
  | 'failed';

export interface RunSummary {
  run_id: string;
  status: RunStatus | string;
  source: string | null;
  model: string | null;
  framework: string | null;
  target_hardware: string | null;
  best_variant: string | null;
  size_delta_pct: number | null;
  latency_ms_delta: number | null;
  start_time: number | null;
}

export interface FeatureSpec {
  name: string;
  dtype?: string;
  choices?: (number | string)[];
}

export interface JobIn {
  model: {
    name?: string;
    ref: string;
    framework: string;
    task: string;
    num_classes?: number;
    target?: string;
    features?: FeatureSpec[] | null;
  };
  test_data: { ref: string };
  target_hardware: string;
  compression?: { methods?: string[] } | null;
  gate?: {
    max_acc_drop?: number;
    max_auc_drop?: number;
    max_rmse_rise?: number;
  } | null;
}

export interface VariantResult {
  model: string;
  kind: string;
  latency_ms: number;
  model_size_kb: number;
  note?: string;
  accuracy?: number;
  auc?: number;
  f1?: number;
  rmse?: number;
  mae?: number;
  speedup_vs_native: number;
}

export interface CompressionReport {
  model: string;
  framework: string;
  task: string;
  target_hardware: string;
  base_model: {
    path: string;
    hash: string;
    size_kb: number;
    mlflow_run_id: string | null;
  };
  best_variant: {
    name: string;
    kind: string;
    format: string;
    size_kb: number;
    note?: string;
  };
  deltas: {
    size_delta_pct: number;
    latency_ms_delta: number;
    speedup_vs_native: number;
    accuracy_delta: number | null;
    auc_delta: number | null;
    f1_delta: number | null;
    rmse_delta: number | null;
  };
  variants: VariantResult[];
  gate_passed: boolean;
  gate_report: any;
}

export interface RegisteredModel {
  name: string;
  production_version: string | null;
  production_run_id: string | null;
  latest_version: string | null;
}

export interface ModelVersion {
  version: string;
  stage: string;
  run_id: string | null;
  status?: string | null;
}
