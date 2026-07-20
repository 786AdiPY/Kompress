// Typed API client. One async function per backend endpoint. OWNED by the
// foundation — page agents import these functions, they must NOT edit this file.
//
// Base URL rule: in dev the Vite server proxies '/api' -> http://localhost:8000,
// so the default relative base works without CORS. Override with VITE_API_BASE.
import type {
  CompressionReport,
  JobIn,
  ModelVersion,
  RegisteredModel,
  RunSummary,
} from './types';

const BASE = import.meta.env.VITE_API_BASE ?? '/api';

// ── response envelope types (server wraps some payloads) ─────────────────────
export interface HealthResponse {
  status: string;
}
export interface SubmitRunResponse {
  run_id: string;
  status: string;
  model: string;
}
export interface RunStatusResponse {
  run_id: string;
  status: string;
}
export interface ApproveResponse {
  approved: true;
  model: string;
  version: string;
  run_id: string;
  stage: string;
  status: string;
}
export interface RejectResponse {
  rejected: true;
  run_id: string;
}
export interface RollbackResponse {
  rolled_back_to: string;
  model: string;
  version: string;
  run_id: string;
  stage: string;
  status: string;
}

// ── shared request helper ────────────────────────────────────────────────────
/** Fetch JSON from the API, throwing an Error carrying the server 'detail' on
 * a non-2xx response. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`.trim();
    try {
      const body = await res.json();
      if (body && typeof body.detail === 'string') detail = body.detail;
      else if (body && body.detail != null) detail = JSON.stringify(body.detail);
    } catch {
      // response had no JSON body; keep the status-line detail
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ── endpoints ────────────────────────────────────────────────────────────────
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

/** Live list of valid target_hardware values for the Submit form. */
export async function getHardwareTargets(): Promise<string[]> {
  const data = await request<{ targets: string[] }>('/hardware-targets');
  return data.targets;
}

export function submitRun(job: JobIn): Promise<SubmitRunResponse> {
  return request<SubmitRunResponse>('/runs', {
    method: 'POST',
    body: JSON.stringify(job),
  });
}

/** The review queue. Pass a status to filter, omit for all UI runs. */
export async function listRuns(status?: string): Promise<RunSummary[]> {
  const qs = status ? `?status=${encodeURIComponent(status)}` : '';
  const data = await request<{ runs: RunSummary[] }>(`/runs${qs}`);
  return data.runs;
}

export function getRun(id: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/runs/${encodeURIComponent(id)}`);
}

export function getReport(id: string): Promise<CompressionReport> {
  return request<CompressionReport>(`/runs/${encodeURIComponent(id)}/report`);
}

/** URL for the winning-variant download. Use as an <a href>, not fetch-json. */
export function artifactUrl(id: string): string {
  return `${BASE}/runs/${encodeURIComponent(id)}/artifact`;
}

export function approveRun(id: string): Promise<ApproveResponse> {
  return request<ApproveResponse>(`/runs/${encodeURIComponent(id)}/approve`, {
    method: 'POST',
  });
}

export function rejectRun(id: string): Promise<RejectResponse> {
  return request<RejectResponse>(`/runs/${encodeURIComponent(id)}/reject`, {
    method: 'POST',
  });
}

export function rollbackRun(
  runId: string,
  toRunId: string,
): Promise<RollbackResponse> {
  return request<RollbackResponse>(
    `/runs/${encodeURIComponent(runId)}/rollback?to_run_id=${encodeURIComponent(toRunId)}`,
    { method: 'POST' },
  );
}

export async function listModels(): Promise<RegisteredModel[]> {
  const data = await request<{ models: RegisteredModel[] }>('/models');
  return data.models;
}

export async function listModelVersions(name: string): Promise<ModelVersion[]> {
  const data = await request<{ versions: ModelVersion[] }>(
    `/models/${encodeURIComponent(name)}/versions`,
  );
  return data.versions;
}
