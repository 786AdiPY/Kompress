// Deployments — the model catalog and rollback controls.
// Route '/deployments'. Lists registered models (GET /models); each row expands
// to its version history (GET /models/{name}/versions), where any non-Production
// version can be re-promoted to Production via rollback.
import { useState } from 'react';
import {
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { listModelVersions, listModels, rollbackRun } from '../api/client';
import type { ModelVersion, RegisteredModel } from '../api/types';
import Loading from '../components/Loading';
import ErrorState from '../components/ErrorState';
import StatusChip from '../components/StatusChip';

const MODEL_COLS = 5;

/** Short, hover-expandable form of a (long) MLflow run id. */
function shortId(id: string | null | undefined): string {
  if (!id) return '—';
  return id.length > 10 ? `${id.slice(0, 8)}…` : id;
}

/** A registered-model catalog row plus its collapsible version-history row. */
function ModelRows({
  model,
  expanded,
  onToggle,
}: {
  model: RegisteredModel;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr>
        <td>{model.name}</td>
        <td>{model.production_version ?? <span className="muted">—</span>}</td>
        <td className="mono" title={model.production_run_id ?? undefined}>
          {shortId(model.production_run_id)}
        </td>
        <td>{model.latest_version ?? <span className="muted">—</span>}</td>
        <td>
          <button
            type="button"
            className="btn"
            onClick={onToggle}
            aria-expanded={expanded}
          >
            {expanded ? 'Hide history' : 'History'}
          </button>
        </td>
      </tr>
      {expanded && (
        <tr>
          <td
            colSpan={MODEL_COLS}
            style={{
              whiteSpace: 'normal',
              background: 'var(--color-surface-2)',
            }}
          >
            <ModelVersions model={model} />
          </td>
        </tr>
      )}
    </>
  );
}

/** The version history for one model, with per-version rollback. */
function ModelVersions({ model }: { model: RegisteredModel }) {
  const queryClient = useQueryClient();

  const {
    data: versions,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ['modelVersions', model.name],
    queryFn: () => listModelVersions(model.name),
  });

  // Per the API, POST /runs/{run_id}/rollback?to_run_id=<id> re-points Production
  // to the target version's run; the path {run_id} is not used for lookup, so we
  // pass the target run_id as both arguments.
  const rollback = useMutation({
    mutationFn: (runId: string) => rollbackRun(runId, runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      queryClient.invalidateQueries({ queryKey: ['modelVersions', model.name] });
    },
  });

  const onRollback = (version: ModelVersion) => {
    if (!version.run_id) return;
    const ok = window.confirm(
      `Roll back Production for "${model.name}" to version ${version.version}` +
        ` (run ${shortId(version.run_id)})?`,
    );
    if (ok) rollback.mutate(version.run_id);
  };

  if (isLoading) return <Loading label="Loading versions…" />;
  if (isError)
    return <ErrorState error={error} title="Failed to load versions" />;
  if (!versions || versions.length === 0)
    return (
      <div className="muted" style={{ padding: 'var(--space-2)' }}>
        No versions registered for this model.
      </div>
    );

  return (
    <div className="stack" style={{ gap: 'var(--space-3)' }}>
      {rollback.isError && (
        <ErrorState error={rollback.error} title="Rollback failed" />
      )}
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Version</th>
              <th>Stage</th>
              <th>Run</th>
              <th>Status</th>
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => {
              const isProduction = v.stage === 'Production';
              const canRollback = !isProduction && !!v.run_id;
              const isThisPending =
                rollback.isPending && rollback.variables === v.run_id;
              return (
                <tr key={v.version}>
                  <td>{v.version}</td>
                  <td>
                    {isProduction ? (
                      <span className="chip chip--approved">Production</span>
                    ) : v.stage && v.stage !== 'None' ? (
                      v.stage
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td className="mono" title={v.run_id ?? undefined}>
                    {shortId(v.run_id)}
                  </td>
                  <td>
                    {v.status ? (
                      <StatusChip status={v.status} />
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    {isProduction ? (
                      <span className="muted">Current</span>
                    ) : canRollback ? (
                      <button
                        type="button"
                        className="btn btn--danger"
                        onClick={() => onRollback(v)}
                        disabled={rollback.isPending}
                      >
                        {isThisPending ? 'Rolling back…' : 'Rollback to this'}
                      </button>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Deployments() {
  const {
    data: models,
    isLoading,
    isError,
    error,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
  });

  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = (name: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  let body: React.ReactNode;
  if (isLoading) {
    body = <Loading label="Loading models…" />;
  } else if (isError) {
    body = <ErrorState error={error} title="Failed to load models" />;
  } else if (!models || models.length === 0) {
    body = <div className="muted">No registered models yet.</div>;
  } else {
    body = (
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Production version</th>
              <th>Production run</th>
              <th>Latest version</th>
              <th aria-label="History" />
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <ModelRows
                key={m.name}
                model={m}
                expanded={expanded.has(m.name)}
                onToggle={() => toggle(m.name)}
              />
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div>
          <h1>Deployments</h1>
          <div className="muted">
            Registered models, their Production version, and rollback controls.
          </div>
        </div>
        <button
          type="button"
          className="btn"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          {isFetching ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      <div className="card">{body}</div>
    </div>
  );
}
