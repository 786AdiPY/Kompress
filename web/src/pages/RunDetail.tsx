// Run detail — "the Plan" / consent screen for one compression run.
// Route: /runs/:runId
//
// Shows the compression report headline (size / latency / quality deltas +
// speedup), provenance, the per-variant benchmark table, the promotion gate
// result, and the Approve / Reject / Download actions. The report may 404 while
// the run is still executing (status pending_gate); we poll getRun until it
// exists and render a friendly waiting state in the meantime.
import { Link, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import {
  approveRun,
  artifactUrl,
  getReport,
  getRun,
  rejectRun,
} from '../api/client';
import type { CompressionReport, VariantResult } from '../api/types';

import DeltaStat from '../components/DeltaStat';
import ErrorState from '../components/ErrorState';
import Loading from '../components/Loading';
import StatusChip from '../components/StatusChip';

import './RunDetail.css';

const POLL_MS = 2500;

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const qc = useQueryClient();

  // Status of the run. Poll while the compression job is still running.
  const runQuery = useQuery({
    queryKey: ['run', runId],
    queryFn: () => getRun(runId as string),
    enabled: !!runId,
    refetchInterval: (query) =>
      query.state.data?.status === 'pending_gate' ? POLL_MS : false,
  });

  const status = runQuery.data?.status;

  // The report ("Plan"). 404s until the job has produced it; keep polling while
  // the run is still in the gate stage, then stop once we have it.
  const reportQuery = useQuery({
    queryKey: ['report', runId],
    queryFn: () => getReport(runId as string),
    enabled: !!runId,
    retry: false,
    refetchInterval: (query) => {
      if (query.state.data) return false;
      return status === 'pending_gate' || status === undefined ? POLL_MS : false;
    },
  });

  const approve = useMutation({
    mutationFn: () => approveRun(runId as string),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['run', runId] });
      qc.invalidateQueries({ queryKey: ['report', runId] });
    },
  });
  const reject = useMutation({
    mutationFn: () => rejectRun(runId as string),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['run', runId] });
      qc.invalidateQueries({ queryKey: ['report', runId] });
    },
  });

  if (!runId) {
    return <ErrorState error="No run id in the URL." title="Invalid run" />;
  }

  return (
    <div>
      <div className="rd-head">
        <div className="rd-head__title">
          <h1>Run</h1>
          <code className="mono muted">{runId}</code>
        </div>
        {status && <StatusChip status={status} />}
        <Link to="/" className="rd-backlink">
          ← All runs
        </Link>
      </div>

      {runQuery.isLoading ? (
        <Loading label="Loading run…" />
      ) : runQuery.isError ? (
        <ErrorState error={runQuery.error} title="Could not load this run" />
      ) : reportQuery.data ? (
        <Plan
          report={reportQuery.data}
          runId={runId}
          status={status}
          approve={approve}
          reject={reject}
        />
      ) : status === 'failed' ? (
        <ErrorState
          title="This run failed"
          error="The compression job failed before producing a report. Check the pipeline logs for this run."
        />
      ) : status === 'rejected' && reportQuery.isError ? (
        <div className="card muted">
          This run was rejected and has no compression report.
        </div>
      ) : status === 'pending_gate' || status === undefined || reportQuery.isFetching ? (
        <div className="card">
          <Loading label="Compression in progress — the plan is not ready yet. Polling…" />
        </div>
      ) : (
        <ErrorState error={reportQuery.error} title="Could not load the report" />
      )}
    </div>
  );
}

// ── the plan (report is available) ───────────────────────────────────────────
type MutationLike = {
  mutate: () => void;
  isPending: boolean;
  isError: boolean;
  error: unknown;
};

function Plan({
  report,
  runId,
  status,
  approve,
  reject,
}: {
  report: CompressionReport;
  runId: string;
  status: string | undefined;
  approve: MutationLike;
  reject: MutationLike;
}) {
  const d = report.deltas;

  // Quality tile: accuracy first, then auc, then rmse (regression).
  const quality = pickQuality(d);

  return (
    <div className="stack">
      {/* ── headline metrics ── */}
      <div className="grid-stats">
        <DeltaStat label="Size" value={d.size_delta_pct} unit="%" goodWhen="negative" />
        <DeltaStat
          label="Latency"
          value={d.latency_ms_delta}
          unit="ms"
          goodWhen="negative"
        />
        <DeltaStat
          label={quality.label}
          value={quality.value}
          goodWhen={quality.goodWhen}
        />
        <SpeedupStat value={d.speedup_vs_native} />
      </div>

      {/* ── provenance ── */}
      <section className="card">
        <div className="rd-section-title">Provenance</div>
        <div className="rd-provenance">
          <Prov label="Model" value={report.model} />
          <Prov label="Framework" value={report.framework} />
          <Prov label="Task" value={report.task} />
          <Prov label="Target hardware" value={report.target_hardware} />
          <Prov label="Base model hash" value={shortHash(report.base_model.hash)} mono />
          <Prov
            label="Best variant"
            value={`${report.best_variant.name} · ${report.best_variant.format}`}
          />
          {report.best_variant.note && (
            <Prov label="Variant note" value={report.best_variant.note} />
          )}
        </div>
      </section>

      {/* ── variants benchmark table ── */}
      <section className="card">
        <div className="rd-section-title">Variants</div>
        <VariantsTable report={report} />
      </section>

      {/* ── promotion gate ── */}
      <section className="card">
        <div className="rd-section-title">Promotion gate</div>
        <GateSection report={report} />
      </section>

      {/* ── actions ── */}
      <section className="card">
        <div className="rd-section-title">Decision</div>
        <Actions
          runId={runId}
          status={status}
          approve={approve}
          reject={reject}
        />
        {approve.isError && (
          <div style={{ marginTop: 'var(--space-3)' }}>
            <ErrorState error={approve.error} title="Approve failed" />
          </div>
        )}
        {reject.isError && (
          <div style={{ marginTop: 'var(--space-3)' }}>
            <ErrorState error={reject.error} title="Reject failed" />
          </div>
        )}
      </section>
    </div>
  );
}

// ── provenance cell ──────────────────────────────────────────────────────────
function Prov({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rd-prov">
      <span className="rd-prov__label">{label}</span>
      <span className={mono ? 'rd-prov__value mono' : 'rd-prov__value'}>{value}</span>
    </div>
  );
}

// ── speedup headline tile (good when > 1, unlike a signed delta) ─────────────
function SpeedupStat({ value }: { value: number | null | undefined }) {
  const tone =
    value == null || Number.isNaN(value) || value === 1
      ? 'neutral'
      : value > 1
        ? 'good'
        : 'bad';
  return (
    <div className="stat">
      <span className="stat__label">Speedup vs native</span>
      <span className={`stat__value stat__value--${tone}`}>
        {value == null || Number.isNaN(value) ? '—' : `${fmt(value, 2)}×`}
      </span>
    </div>
  );
}

// ── variants table ───────────────────────────────────────────────────────────
type MetricKey = 'accuracy' | 'auc' | 'f1' | 'rmse' | 'mae';
const CLASS_METRICS: { key: MetricKey; label: string; digits: number }[] = [
  { key: 'accuracy', label: 'Accuracy', digits: 4 },
  { key: 'auc', label: 'AUC', digits: 4 },
  { key: 'f1', label: 'F1', digits: 4 },
];
const REG_METRICS: { key: MetricKey; label: string; digits: number }[] = [
  { key: 'rmse', label: 'RMSE', digits: 2 },
  { key: 'mae', label: 'MAE', digits: 2 },
];

function VariantsTable({ report }: { report: CompressionReport }) {
  const metricCols = report.task === 'regression' ? REG_METRICS : CLASS_METRICS;
  const bestName = report.best_variant.name;

  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Variant</th>
            <th className="rd-num">Latency (ms)</th>
            <th className="rd-num">Size (KB)</th>
            <th className="rd-num">Speedup</th>
            {metricCols.map((m) => (
              <th key={m.key} className="rd-num">
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {report.variants.map((v: VariantResult) => {
            const isBest = v.model === bestName;
            const isNative = v.kind === 'native';
            return (
              <tr key={v.model} className={isBest ? 'rd-best' : undefined}>
                <td>
                  {v.model}
                  {isBest && <span className="rd-tag rd-tag--best">best</span>}
                  {isNative && <span className="rd-tag">baseline</span>}
                </td>
                <td className="rd-num">{fmt(v.latency_ms, 3)}</td>
                <td className="rd-num">{fmt(v.model_size_kb, 1)}</td>
                <td className="rd-num">{fmt(v.speedup_vs_native, 2)}×</td>
                {metricCols.map((m) => (
                  <td key={m.key} className="rd-num">
                    {fmt(v[m.key], m.digits)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── promotion gate ───────────────────────────────────────────────────────────
interface GateCheck {
  variant?: string;
  passed?: boolean;
  latency_ms?: number;
  speedup?: number;
  acc_drop?: number;
  auc_drop?: number;
  rmse_rise?: number;
}

function GateSection({ report }: { report: CompressionReport }) {
  const passed = report.gate_passed;
  const gr = report.gate_report ?? {};
  const checks: GateCheck[] = Array.isArray(gr.checks) ? gr.checks : [];
  const isRegression = report.task === 'regression';

  return (
    <div>
      <div className={`rd-gate ${passed ? 'rd-gate--pass' : 'rd-gate--fail'}`}>
        <span className="rd-gate__dot" aria-hidden="true" />
        {passed ? 'Gate passed' : 'Gate failed'}
      </div>

      {checks.length > 0 ? (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Variant</th>
                <th>Result</th>
                <th className="rd-num">Latency (ms)</th>
                <th className="rd-num">Speedup</th>
                {isRegression ? (
                  <th className="rd-num">RMSE rise</th>
                ) : (
                  <>
                    <th className="rd-num">Acc drop</th>
                    <th className="rd-num">AUC drop</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {checks.map((c, i) => (
                <tr key={c.variant ?? i}>
                  <td>{c.variant ?? '—'}</td>
                  <td>
                    <span className={c.passed ? 'rd-pass' : 'rd-fail'}>
                      {c.passed ? 'Pass' : 'Fail'}
                    </span>
                  </td>
                  <td className="rd-num">{fmt(c.latency_ms, 3)}</td>
                  <td className="rd-num">{fmt(c.speedup, 2)}×</td>
                  {isRegression ? (
                    <td className="rd-num">{fmt(c.rmse_rise, 4)}</td>
                  ) : (
                    <>
                      <td className="rd-num">{fmt(c.acc_drop, 4)}</td>
                      <td className="rd-num">{fmt(c.auc_drop, 4)}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="muted">No per-variant gate checks reported.</div>
      )}
    </div>
  );
}

// ── actions bar ──────────────────────────────────────────────────────────────
function Actions({
  runId,
  status,
  approve,
  reject,
}: {
  runId: string;
  status: string | undefined;
  approve: MutationLike;
  reject: MutationLike;
}) {
  const canDecide = status === 'pending_approval';
  const busy = approve.isPending || reject.isPending;

  return (
    <div className="rd-actions">
      <button
        type="button"
        className="btn btn--primary"
        disabled={!canDecide || busy}
        onClick={() => approve.mutate()}
      >
        {approve.isPending ? 'Approving…' : 'Approve'}
      </button>
      <button
        type="button"
        className="btn btn--danger"
        disabled={!canDecide || busy}
        onClick={() => reject.mutate()}
      >
        {reject.isPending ? 'Rejecting…' : 'Reject'}
      </button>

      {!canDecide && (
        <span className="muted">
          {status === 'approved'
            ? 'This run is already approved.'
            : status === 'rejected'
              ? 'This run was rejected.'
              : 'Approval is available once the run is pending approval.'}
        </span>
      )}

      <a
        className="btn rd-actions__spacer"
        href={artifactUrl(runId)}
        target="_blank"
        rel="noreferrer"
      >
        Download compressed model
      </a>
    </div>
  );
}

// ── helpers ──────────────────────────────────────────────────────────────────
function pickQuality(d: CompressionReport['deltas']): {
  label: string;
  value: number | null;
  goodWhen: 'positive' | 'negative';
} {
  if (d.accuracy_delta != null) {
    return { label: 'Accuracy', value: d.accuracy_delta, goodWhen: 'positive' };
  }
  if (d.auc_delta != null) {
    return { label: 'AUC', value: d.auc_delta, goodWhen: 'positive' };
  }
  if (d.rmse_delta != null) {
    return { label: 'RMSE', value: d.rmse_delta, goodWhen: 'negative' };
  }
  return { label: 'Quality', value: null, goodWhen: 'positive' };
}

function shortHash(hash: string | null | undefined): string {
  if (!hash) return '—';
  return hash.length > 16 ? `${hash.slice(0, 12)}…` : hash;
}

function fmt(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return '—';
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}
