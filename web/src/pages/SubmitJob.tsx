// Submit a compression job (route '/submit').
//
// The form assembles a `JobIn` and POSTs it via `submitRun`. The API accepts
// POINTERS ONLY (s3:// / mlflow:// URIs) — never raw file bytes — so the UI
// mirrors that boundary and forwards the server's own error `detail` verbatim
// (e.g. 400 "must be a remote pointer", 501 "scheme not implemented").
import { useState } from 'react';
import type { FormEvent } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';

import { getHardwareTargets, submitRun } from '../api/client';
import type { JobIn } from '../api/types';
import ErrorState from '../components/ErrorState';
import './SubmitJob.css';

const FRAMEWORKS = ['xgboost', 'lightgbm', 'sklearn', 'pytorch'] as const;
const TASKS = [
  'binary_classification',
  'multiclass_classification',
  'regression',
] as const;

export default function SubmitJob() {
  const navigate = useNavigate();

  // ── core fields ────────────────────────────────────────────────────────────
  const [name, setName] = useState('');
  const [modelRef, setModelRef] = useState('');
  const [testRef, setTestRef] = useState('');
  const [framework, setFramework] = useState<string>(FRAMEWORKS[0]);
  const [task, setTask] = useState<string>(TASKS[0]);
  const [numClasses, setNumClasses] = useState('3');
  const [target, setTarget] = useState('target');
  const [hardware, setHardware] = useState('');

  // ── advanced (gate + compression) ───────────────────────────────────────────
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxAccDrop, setMaxAccDrop] = useState('0.01');
  const [maxAucDrop, setMaxAucDrop] = useState('0.01');
  const [maxRmseRise, setMaxRmseRise] = useState('0.05');
  const [methods, setMethods] = useState('');

  const isMulticlass = task === 'multiclass_classification';

  const hardwareQuery = useQuery({
    queryKey: ['hardware-targets'],
    queryFn: getHardwareTargets,
  });

  const mutation = useMutation({
    mutationFn: submitRun,
    onSuccess: (res) => navigate(`/runs/${res.run_id}`),
  });

  // ── validation: enough to enable submit; the server is the source of truth ──
  const numClassesValid =
    !isMulticlass || Number.isFinite(Number(numClasses)) && Number(numClasses) >= 2;
  const canSubmit =
    modelRef.trim() !== '' &&
    testRef.trim() !== '' &&
    hardware !== '' &&
    numClassesValid &&
    !mutation.isPending;

  function buildJob(): JobIn {
    const model: JobIn['model'] = {
      ref: modelRef.trim(),
      framework,
      task,
    };
    if (name.trim()) model.name = name.trim();
    if (target.trim()) model.target = target.trim();
    if (isMulticlass) {
      const n = parseInt(numClasses, 10);
      if (Number.isFinite(n)) model.num_classes = n;
    }

    const job: JobIn = {
      model,
      test_data: { ref: testRef.trim() },
      target_hardware: hardware,
    };

    const methodList = methods
      .split(',')
      .map((m) => m.trim())
      .filter(Boolean);
    if (methodList.length) job.compression = { methods: methodList };

    const gate: NonNullable<JobIn['gate']> = {};
    const acc = parseFloat(maxAccDrop);
    const auc = parseFloat(maxAucDrop);
    const rmse = parseFloat(maxRmseRise);
    if (Number.isFinite(acc)) gate.max_acc_drop = acc;
    if (Number.isFinite(auc)) gate.max_auc_drop = auc;
    if (Number.isFinite(rmse)) gate.max_rmse_rise = rmse;
    if (Object.keys(gate).length) job.gate = gate;

    return job;
  }

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!canSubmit) return;
    mutation.mutate(buildJob());
  }

  return (
    <div className="stack">
      <div>
        <h1 className="page-title">Submit a compression job</h1>
        <p className="muted">
          Point the pipeline at an existing model and test set. It compresses,
          benchmarks on your target hardware, and gates the result for review.
        </p>
      </div>

      <form className="card submit-form" onSubmit={onSubmit} noValidate>
        <div className="submit-form__grid">
          {/* ── model identity & pointers ─────────────────────────────────── */}
          <div className="field field--full">
            <label htmlFor="model-name">
              Model name <span className="field__hint">(optional)</span>
            </label>
            <input
              id="model-name"
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="fraud-scorer (defaults to the framework name)"
            />
          </div>

          <div className="field field--full">
            <label htmlFor="model-ref">Model pointer</label>
            <input
              id="model-ref"
              className="input mono"
              value={modelRef}
              onChange={(e) => setModelRef(e.target.value)}
              placeholder="s3://bucket/model.pkl — pointer, not upload"
              required
            />
          </div>

          <div className="field field--full">
            <label htmlFor="test-ref">Test data pointer</label>
            <input
              id="test-ref"
              className="input mono"
              value={testRef}
              onChange={(e) => setTestRef(e.target.value)}
              placeholder="s3://bucket/test.csv — pointer, not upload"
              required
            />
          </div>

          <p className="form-note form-note--boundary field--full">
            Pointers only (S3 / MLflow URI). Files are never uploaded.
          </p>

          {/* ── framework / task / classes / target / hardware ─────────────── */}
          <div className="field">
            <label htmlFor="framework">Framework</label>
            <select
              id="framework"
              className="select"
              value={framework}
              onChange={(e) => setFramework(e.target.value)}
            >
              {FRAMEWORKS.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="task">Task</label>
            <select
              id="task"
              className="select"
              value={task}
              onChange={(e) => setTask(e.target.value)}
            >
              {TASKS.map((t) => (
                <option key={t} value={t}>
                  {t.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </div>

          {isMulticlass && (
            <div className="field">
              <label htmlFor="num-classes">Number of classes</label>
              <input
                id="num-classes"
                className="input"
                type="number"
                min={2}
                step={1}
                value={numClasses}
                onChange={(e) => setNumClasses(e.target.value)}
              />
            </div>
          )}

          <div className="field">
            <label htmlFor="target">
              Target column <span className="field__hint">(label to predict)</span>
            </label>
            <input
              id="target"
              className="input"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="target"
            />
          </div>

          <div className="field field--full">
            <label htmlFor="hardware">Target hardware</label>
            <select
              id="hardware"
              className="select"
              value={hardware}
              onChange={(e) => setHardware(e.target.value)}
              disabled={hardwareQuery.isLoading || hardwareQuery.isError}
              required
            >
              <option value="" disabled>
                {hardwareQuery.isLoading
                  ? 'Loading hardware targets…'
                  : 'Select target hardware…'}
              </option>
              {(hardwareQuery.data ?? []).map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
            {hardwareQuery.isError && (
              <span className="form-note" style={{ color: 'var(--color-bad)' }}>
                Could not load hardware targets — is the API running?
              </span>
            )}
          </div>
        </div>

        {/* ── advanced (collapsible) ───────────────────────────────────────── */}
        <div className="advanced">
          <button
            type="button"
            className="advanced__toggle"
            aria-expanded={showAdvanced}
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? '▾ Advanced' : '▸ Advanced'}
          </button>

          {showAdvanced && (
            <div className="advanced__body">
              <p className="form-note" style={{ marginBottom: 'var(--space-4)' }}>
                Quality gate thresholds — a variant is auto-rejected if it drops
                below these. Leave the defaults unless you have a reason.
              </p>
              <div className="submit-form__grid">
                <div className="field">
                  <label htmlFor="max-acc-drop">Max accuracy drop</label>
                  <input
                    id="max-acc-drop"
                    className="input"
                    type="number"
                    step="0.01"
                    value={maxAccDrop}
                    onChange={(e) => setMaxAccDrop(e.target.value)}
                  />
                </div>
                <div className="field">
                  <label htmlFor="max-auc-drop">Max AUC drop</label>
                  <input
                    id="max-auc-drop"
                    className="input"
                    type="number"
                    step="0.01"
                    value={maxAucDrop}
                    onChange={(e) => setMaxAucDrop(e.target.value)}
                  />
                </div>
                <div className="field">
                  <label htmlFor="max-rmse-rise">Max RMSE rise</label>
                  <input
                    id="max-rmse-rise"
                    className="input"
                    type="number"
                    step="0.01"
                    value={maxRmseRise}
                    onChange={(e) => setMaxRmseRise(e.target.value)}
                  />
                </div>
                <div className="field field--full">
                  <label htmlFor="methods">
                    Compression methods{' '}
                    <span className="field__hint">
                      (optional, comma-separated)
                    </span>
                  </label>
                  <input
                    id="methods"
                    className="input mono"
                    value={methods}
                    onChange={(e) => setMethods(e.target.value)}
                    placeholder="quantization, pruning, onnx"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {mutation.isError && (
          <div style={{ marginTop: 'var(--space-4)' }}>
            <ErrorState error={mutation.error} title="Submission failed" />
          </div>
        )}

        <div className="submit-actions">
          <button
            type="submit"
            className="btn btn--primary"
            disabled={!canSubmit}
          >
            {mutation.isPending ? 'Submitting…' : 'Submit job'}
          </button>
          {!canSubmit && !mutation.isPending && (
            <span className="form-note">
              Model pointer, test data pointer, and target hardware are required.
            </span>
          )}
        </div>
      </form>
    </div>
  );
}
