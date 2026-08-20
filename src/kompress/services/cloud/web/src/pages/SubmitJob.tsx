// Submit a compression job (route '/submit').
//
// The user picks a .pkl model file and a test-data file from disk. Each is
// streamed to POST /uploads as soon as it's picked, which stores it in
// object storage and hands back an s3:// pointer; the form assembles a
// `JobIn` from those pointers and POSTs it via `submitRun`. The engine
// itself still only ever sees pointers — this just does the pointer-minting
// upload for the user instead of asking them to host the file themselves.
import { useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';

import { getHardwareTargets, submitRun, uploadFile } from '../api/client';
import type { UploadResponse } from '../api/client';
import type { JobIn } from '../api/types';
import ErrorState from '../components/ErrorState';
import './SubmitJob.css';

const FRAMEWORKS = ['xgboost', 'lightgbm', 'sklearn', 'pytorch'] as const;
const TASKS = [
  'binary_classification',
  'multiclass_classification',
  'regression',
] as const;

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let n = bytes / 1024;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(1)} ${units[i]}`;
}

/** One pick-a-file-and-upload-it field. Uploads immediately on selection and
 * exposes the resulting pointer via `onUploaded`. */
function FileUploadField({
  id,
  label,
  accept,
  hint,
  kind,
  onUploaded,
}: {
  id: string;
  label: string;
  accept: string;
  hint: string;
  kind: 'model' | 'test_data';
  onUploaded: (res: UploadResponse | null) => void;
}) {
  const [fileName, setFileName] = useState('');
  const [fileSize, setFileSize] = useState(0);

  const upload = useMutation({
    mutationFn: (file: File) => uploadFile(file, kind),
    onSuccess: (res) => onUploaded(res),
    onError: () => onUploaded(null),
  });

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ''; // allow re-picking the same filename after an error
    if (!file) return;
    setFileName(file.name);
    setFileSize(file.size);
    onUploaded(null);
    upload.mutate(file);
  }

  return (
    <div className="field field--full">
      <label htmlFor={id}>{label}</label>
      <div className="file-drop">
        <label htmlFor={id} className="file-drop__button">
          Choose file…
        </label>
        <input
          id={id}
          type="file"
          accept={accept}
          className="file-drop__input"
          onChange={onChange}
        />
        <span className="file-drop__status">
          {!fileName && hint}
          {fileName && upload.isPending && `Uploading ${fileName}…`}
          {fileName && upload.isSuccess && (
            <span className="file-drop__ok">
              ✓ {fileName} ({humanSize(fileSize)})
            </span>
          )}
          {fileName && upload.isError && (
            <span className="file-drop__err">
              {(upload.error as Error).message}
            </span>
          )}
        </span>
      </div>
    </div>
  );
}

export default function SubmitJob() {
  const navigate = useNavigate();

  // ── core fields ────────────────────────────────────────────────────────────
  const [name, setName] = useState('');
  const [modelUpload, setModelUpload] = useState<UploadResponse | null>(null);
  const [testUpload, setTestUpload] = useState<UploadResponse | null>(null);
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
    modelUpload != null &&
    testUpload != null &&
    hardware !== '' &&
    numClassesValid &&
    !mutation.isPending;

  function buildJob(): JobIn {
    if (!modelUpload || !testUpload) throw new Error('files not uploaded yet');
    const model: JobIn['model'] = {
      ref: modelUpload.ref,
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
      test_data: { ref: testUpload.ref },
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
          Upload a model and test set. It compresses, benchmarks on your
          target hardware, and gates the result for review.
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

          <FileUploadField
            id="model-file"
            label="Model file"
            accept=".pkl,.pickle"
            hint="Choose a .pkl model file"
            kind="model"
            onUploaded={setModelUpload}
          />

          <FileUploadField
            id="test-file"
            label="Test data file"
            accept=".csv"
            hint="Choose a .csv test set"
            kind="test_data"
            onUploaded={setTestUpload}
          />

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
              Model file, test data file, and target hardware are required.
            </span>
          )}
        </div>
      </form>
    </div>
  );
}
