// Typed API client. One async function per backend endpoint. OWNED by the
// foundation — page agents import these functions, they must NOT edit this file.
//
// Base URL rule: in dev the Vite server proxies '/api' -> http://localhost:8000,
// so the default relative base works without CORS. Override with VITE_API_BASE.
import type {
  CompressionReport,
  ExportFormat,
  JobIn,
  ModelVersion,
  ProgressReport,
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

export interface RecommendationResponse {
  framework: string;
  task: string;
  recommended_methods: string[];
  quality_tolerance: 'strict' | 'standard' | 'relaxed';
  reasoning: string;
  context_summary?: string;
  ai_powered: boolean;
}

export async function getAiRecommendation(params: {
  modelName?: string;
  framework?: string;
  targetHardware?: string;
  contextText?: string;
  contextFile?: File | null;
}): Promise<RecommendationResponse> {
  const form = new FormData();
  if (params.modelName) form.append('model_name', params.modelName);
  if (params.framework) form.append('framework', params.framework);
  if (params.targetHardware) form.append('target_hardware', params.targetHardware);
  if (params.contextText) form.append('context_text', params.contextText);
  if (params.contextFile) form.append('context_file', params.contextFile);

  const res = await fetch(`${BASE}/recommend`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    throw new Error(`AI recommendation failed (${res.status})`);
  }
  return res.json();
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

export interface UploadResponse {
  ref: string;
  filename: string;
  kind: string;
}

/** Stream a raw file to the server, which stores it in object storage and
 * hands back an s3:// pointer — the pointer is what POST /runs consumes.
 * Bypasses the shared `request` helper: it forces a JSON content-type, but
 * a multipart body needs the browser to set its own (with the boundary). */
export async function uploadFile(
  file: File,
  kind: 'model' | 'test_data',
): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(
    `${BASE}/uploads?kind=${encodeURIComponent(kind)}`,
    { method: 'POST', body: form },
  );
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`.trim();
    try {
      const body = await res.json();
      if (body && typeof body.detail === 'string') detail = body.detail;
      else if (body && body.detail != null) detail = JSON.stringify(body.detail);
    } catch {
      // response had no JSON body; keep the status-line detail
    }
    throw new Error(detail || `Upload failed (${res.status})`);
  }
  return (await res.json()) as UploadResponse;
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

export async function getReport(
  id: string,
): Promise<CompressionReport | ProgressReport | null> {
  const res = await fetch(`${BASE}/runs/${encodeURIComponent(id)}/report`, {
    headers: { 'Content-Type': 'application/json' },
  });

  // 202 = compression still in progress, return live progress payload
  if (res.status === 202) {
    try {
      const body = await res.json();
      return {
        is_partial: true,
        status: body.run_status || body.status || 'pending_gate',
        stage: body.stage ?? 1,
        stage_name: body.stage_name || 'Optimizing model graph',
        model_name: body.model_name,
        target_hardware: body.target_hardware,
        variants_done: body.variants_done || [],
        message: body.message,
      };
    } catch {
      return {
        is_partial: true,
        status: 'pending_gate',
        stage: 1,
        stage_name: 'Optimizing model graph',
      };
    }
  }

  // 404 = run doesn't exist or report will never exist (legacy fallback)
  if (res.status === 404) return null;

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`.trim();
    try {
      const body = await res.json();
      if (body && typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* no json body */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }

  return (await res.json()) as CompressionReport;
}

/** URL for the winning-variant download. Use as an <a href>, not fetch-json. */
export function artifactUrl(id: string): string {
  return `${BASE}/runs/${encodeURIComponent(id)}/artifact`;
}

/** Device export targets for the 'Download for device' UI, flagged with which
 * one is recommended for the run's target hardware. */
export async function getExportFormats(hardware?: string | null): Promise<ExportFormat[]> {
  const qs = hardware ? `?hardware=${encodeURIComponent(hardware)}` : '';
  const data = await request<{ formats: ExportFormat[] }>(`/export-formats${qs}`);
  return data.formats;
}

/** URL for a device-specific export download. Use as an <a href>, not fetch-json —
 * the server converts ONNX -> the target format and streams the file back. */
export function exportUrl(id: string, format: string): string {
  return `${BASE}/runs/${encodeURIComponent(id)}/export?format=${encodeURIComponent(format)}`;
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
