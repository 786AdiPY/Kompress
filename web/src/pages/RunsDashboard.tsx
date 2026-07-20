// RunsDashboard — the review queue (route '/').
//
// Live-updating table of self-serve compression runs, filterable by status.
// Every row deep-links to /runs/{run_id}. Data flows exclusively through the
// typed client (listRuns) via react-query; this page holds no fetch logic of
// its own. Owned by the Dashboard page agent — safe to overwrite in full.
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';

import { listRuns } from '../api/client';
import type { RunStatus, RunSummary } from '../api/types';
import StatusChip from '../components/StatusChip';
import Loading from '../components/Loading';
import ErrorState from '../components/ErrorState';
import './RunsDashboard.css';

// Filter tabs. `undefined` value => the "All" tab (no ?status= filter).
const TABS: { label: string; value?: RunStatus }[] = [
  { label: 'All' },
  { label: 'Pending Gate', value: 'pending_gate' },
  { label: 'Pending Approval', value: 'pending_approval' },
  { label: 'Approved', value: 'approved' },
  { label: 'Rejected', value: 'rejected' },
  { label: 'Failed', value: 'failed' },
];

export default function RunsDashboard() {
  const [status, setStatus] = useState<RunStatus | undefined>(undefined);
  const navigate = useNavigate();

  const { data, isLoading, isError, error, isFetching } = useQuery({
    queryKey: ['runs', status],
    queryFn: () => listRuns(status),
    refetchInterval: 4000, // live-update the review queue
  });

  const runs = data ?? [];

  return (
    <div>
      <header className="dashboard-header">
        <div className="dashboard-header__title">
          <h1>Runs</h1>
          <span className="muted">
            The review queue for self-serve compression jobs.
          </span>
        </div>
        <Link to="/submit" className="btn btn--primary" data-tour="submit-job">
          Submit new job
        </Link>
      </header>

      <nav className="tabs" aria-label="Filter runs by status" data-tour="status-tabs">
        {TABS.map((tab) => {
          const active = tab.value === status;
          return (
            <button
              key={tab.label}
              type="button"
              className={active ? 'tab tab--active' : 'tab'}
              aria-pressed={active}
              onClick={() => setStatus(tab.value)}
            >
              {tab.label}
            </button>
          );
        })}
      </nav>

      <div className="card" data-tour="runs-table">
        {isLoading ? (
          <Loading label="Loading runs…" />
        ) : isError ? (
          <ErrorState error={error} title="Could not load runs" />
        ) : runs.length === 0 ? (
          <EmptyState />
        ) : (
          <RunsTable runs={runs} onOpen={(id) => navigate(`/runs/${id}`)} />
        )}

        {!isLoading && !isError && (
          <p className="live-note muted" aria-live="polite">
            <span
              className={isFetching ? 'live-dot live-dot--active' : 'live-dot'}
              aria-hidden="true"
            />
            {isFetching ? 'Refreshing…' : 'Live — refreshes every few seconds'}
          </p>
        )}
      </div>
    </div>
  );
}

// ── table ────────────────────────────────────────────────────────────────────
function RunsTable({
  runs,
  onOpen,
}: {
  runs: RunSummary[];
  onOpen: (runId: string) => void;
}) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Framework</th>
            <th>Hardware</th>
            <th>Best Variant</th>
            <th>Size Δ</th>
            <th>Latency Δ</th>
            <th>Status</th>
            <th>Submitted</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.run_id}
              className="run-row"
              onClick={() => onOpen(run.run_id)}
            >
              <td>
                {/* Keyboard-accessible entry point; the whole row is also
                    clickable for pointer users. */}
                <Link
                  to={`/runs/${run.run_id}`}
                  onClick={(e) => e.stopPropagation()}
                >
                  {run.model ?? 'Untitled model'}
                </Link>
                <div className="run-row__id mono">{shortId(run.run_id)}</div>
              </td>
              <td>{dash(run.framework)}</td>
              <td>{dash(run.target_hardware)}</td>
              <td>{dash(run.best_variant)}</td>
              <td>
                <DeltaCell value={run.size_delta_pct} unit="%" />
              </td>
              <td>
                <DeltaCell value={run.latency_ms_delta} unit="ms" />
              </td>
              <td>
                <StatusChip status={run.status} />
              </td>
              <td className="muted">{formatTime(run.start_time)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── empty state ───────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-state__title">No runs yet</div>
      <p>
        Submit a compression job to get started —{' '}
        <Link to="/submit">Submit one</Link>.
      </p>
    </div>
  );
}

// ── delta cell (DeltaStat-style coloring, inline for table density) ───────────
// Sign convention: for size % and latency ms, negative = good (smaller/faster).
function DeltaCell({ value, unit }: { value: number | null; unit: string }) {
  if (value == null || Number.isNaN(value)) {
    return <span className="delta-cell stat__value--neutral">—</span>;
  }
  const tone: 'good' | 'bad' | 'neutral' =
    value === 0 ? 'neutral' : value < 0 ? 'good' : 'bad';
  const rounded =
    Math.abs(value) >= 100 ? Math.round(value) : Math.round(value * 100) / 100;
  const sign = rounded > 0 ? '+' : '';
  return (
    <span className={`delta-cell stat__value--${tone}`}>
      {sign}
      {rounded}
      {unit}
    </span>
  );
}

// ── formatting helpers ────────────────────────────────────────────────────────
function dash(value: string | null | undefined): string {
  return value == null || value === '' ? '—' : value;
}

function shortId(runId: string): string {
  return runId.length > 10 ? runId.slice(0, 10) : runId;
}

/** start_time is an epoch-milliseconds timestamp (nullable). */
function formatTime(ms: number | null): string {
  if (ms == null) return '—';
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}
