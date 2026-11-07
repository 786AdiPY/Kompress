// Submit a compression job (route '/submit').
import { useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { FileText } from 'lucide-react';

import { getHardwareTargets, submitRun, uploadFile, getAiRecommendation } from '../api/client';
import type { UploadResponse, RecommendationResponse } from '../api/client';
import type { JobIn } from '../api/types';
import ErrorState from '../components/ErrorState';
import './SubmitJob.css';

const FRAMEWORKS = ['pytorch', 'xgboost', 'lightgbm', 'sklearn', 'onnx'] as const;
const TASKS = [
  'binary_classification',
  'multiclass_classification',
  'regression',
] as const;

const ALL_COMPRESSORS = [
  { id: 'onnx_fp32', label: 'ONNX FP32 (Portable Baseline)' },
  { id: 'onnx_fp16', label: 'ONNX FP16 (Half-Precision)' },
  { id: 'onnx_fp16_mixed', label: 'ONNX FP16 Mixed Precision' },
  { id: 'onnx_fp16_adaptive', label: 'ONNX FP16 Adaptive' },
  { id: 'onnx_int8_dynamic', label: 'ONNX INT8 Dynamic Quantization' },
  { id: 'onnx_int8_static', label: 'ONNX INT8 Static Quantization' },
  { id: 'trt_int8', label: 'TensorRT INT8 GPU Engine' },
];

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

function detectFramework(fileName: string): string {
  const ext = fileName.split('.').pop()?.toLowerCase();
  if (['pt', 'pth', 'safetensors', 'bin'].includes(ext ?? '')) return 'pytorch';
  if (['pkl', 'pickle', 'joblib'].includes(ext ?? '')) return 'sklearn';
  if (['onnx'].includes(ext ?? '')) return 'onnx';
  return 'pytorch';
}

function detectTask(fileName: string): string {
  const name = fileName.toLowerCase();
  if (name.includes('regress') || name.includes('price') || name.includes('house') || name.includes('val')) {
    return 'regression';
  }
  if (name.includes('multi') || name.includes('cifar') || name.includes('mnist') || name.includes('image')) {
    return 'multiclass_classification';
  }
  return 'binary_classification';
}

/** One pick-a-file-and-upload-it field. */
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
  onUploaded: (res: UploadResponse | null, name: string) => void;
}) {
  const [fileName, setFileName] = useState('');
  const [fileSize, setFileSize] = useState(0);

  const upload = useMutation({
    mutationFn: (file: File) => uploadFile(file, kind),
    onSuccess: (res) => onUploaded(res, fileName),
    onError: () => onUploaded(null, ''),
  });

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setFileName(file.name);
    setFileSize(file.size);
    onUploaded(null, file.name);
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

  // ── core state ────────────────────────────────────────────────────────────
  const [name, setName] = useState('');
  const [modelUpload, setModelUpload] = useState<UploadResponse | null>(null);
  const [modelFileName, setModelFileName] = useState('');
  const [testUpload, setTestUpload] = useState<UploadResponse | null>(null);
  const [testFileName, setTestFileName] = useState('');

  // Context & AI Recommendation state
  const [contextText, setContextText] = useState('');
  const [contextFile, setContextFile] = useState<File | null>(null);
  const [contextFileName, setContextFileName] = useState('');
  const [aiRecommendation, setAiRecommendation] = useState<RecommendationResponse | null>(null);
  const [isRecommending, setIsRecommending] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);

  // Framework & Task (hidden on main form, present in advanced)
  const [framework, setFramework] = useState<string>('pytorch');
  const [task, setTask] = useState<string>('binary_classification');

  const [numClasses, setNumClasses] = useState('3');
  const [target, setTarget] = useState('target');
  const [hardware, setHardware] = useState('');

  // Quality tolerance control
  const [tolerancePreset, setTolerancePreset] = useState<'strict' | 'standard' | 'relaxed' | 'custom'>('standard');
  const [customTolerancePct, setCustomTolerancePct] = useState('1.0');

  // AI strategy vs Compress all algorithms
  const [compressionStrategy, setCompressionStrategy] = useState<'ai' | 'all' | 'custom'>('ai');
  const [selectedMethods, setSelectedMethods] = useState<string[]>([]);

  // Advanced section toggle
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxAccDrop, setMaxAccDrop] = useState('0.01');
  const [maxAucDrop, setMaxAucDrop] = useState('0.01');
  const [maxRmseRise, setMaxRmseRise] = useState('0.05');

  // Tabular detection
  const isTabular =
    framework === 'sklearn' ||
    framework === 'xgboost' ||
    framework === 'lightgbm' ||
    testFileName.toLowerCase().endsWith('.csv');

  const isMulticlass = task === 'multiclass_classification';

  const hardwareQuery = useQuery({
    queryKey: ['hardware-targets'],
    queryFn: getHardwareTargets,
  });

  const mutation = useMutation({
    mutationFn: submitRun,
    onSuccess: (res) => navigate(`/runs/${res.run_id}`),
  });

  // Handle Model Upload callback -> trigger auto-detection
  function handleModelUploaded(res: UploadResponse | null, fname: string) {
    setModelUpload(res);
    if (fname) {
      setModelFileName(fname);
      const autoFw = detectFramework(fname);
      setFramework(autoFw);
      const autoTask = detectTask(fname);
      setTask(autoTask);
    }
  }

  // Handle Test Set Upload callback
  function handleTestUploaded(res: UploadResponse | null, fname: string) {
    setTestUpload(res);
    if (fname) {
      setTestFileName(fname);
      if (modelFileName) {
        const autoTask = detectTask(modelFileName + ' ' + fname);
        setTask(autoTask);
      }
    }
  }

  // Handle Context File selection (.pdf, .docx, .md, .txt)
  function handleContextFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      setContextFile(file);
      setContextFileName(file.name);
    }
  }

  // Invoke Groq AI Recommendation
  async function handleGetAiRecommendation() {
    setIsRecommending(true);
    setRecError(null);
    try {
      const rec = await getAiRecommendation({
        modelName: name || modelFileName,
        framework,
        targetHardware: hardware || 'cpu-generic',
        contextText,
        contextFile,
      });
      setAiRecommendation(rec);
      if (rec.task) setTask(rec.task);
      if (rec.framework) setFramework(rec.framework);
      if (rec.quality_tolerance) setTolerancePreset(rec.quality_tolerance);
      if (rec.recommended_methods && rec.recommended_methods.length > 0) {
        setSelectedMethods(rec.recommended_methods);
        setCompressionStrategy('ai');
      }
    } catch (err) {
      setRecError((err as Error).message);
    } finally {
      setIsRecommending(false);
    }
  }

  // Validation
  const numClassesValid =
    !isMulticlass || (Number.isFinite(Number(numClasses)) && Number(numClasses) >= 2);
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
    if (isTabular && target.trim()) model.target = target.trim();
    if (isMulticlass) {
      const n = parseInt(numClasses, 10);
      if (Number.isFinite(n)) model.num_classes = n;
    }

    const job: JobIn = {
      model,
      test_data: { ref: testUpload.ref },
      target_hardware: hardware,
    };

    if (compressionStrategy === 'custom' && selectedMethods.length > 0) {
      job.compression = { methods: selectedMethods };
    } else {
      job.compression = null; // AI / Bypass runs all hardware algorithms
    }

    const gate: NonNullable<JobIn['gate']> = {};
    let tolVal = 0.01;
    if (tolerancePreset === 'strict') tolVal = 0.005;
    else if (tolerancePreset === 'relaxed') tolVal = 0.05;
    else if (tolerancePreset === 'custom') {
      const p = parseFloat(customTolerancePct);
      if (Number.isFinite(p) && p >= 0) tolVal = p / 100.0;
    }

    if (showAdvanced) {
      const acc = parseFloat(maxAccDrop);
      const auc = parseFloat(maxAucDrop);
      const rmse = parseFloat(maxRmseRise);
      if (Number.isFinite(acc)) gate.max_acc_drop = acc;
      if (Number.isFinite(auc)) gate.max_auc_drop = auc;
      if (Number.isFinite(rmse)) gate.max_rmse_rise = rmse;
    } else {
      gate.max_acc_drop = tolVal;
      gate.max_auc_drop = tolVal;
      gate.max_rmse_rise = tolVal * 5.0;
    }
    job.gate = gate;

    return job;
  }

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!canSubmit) return;
    mutation.mutate(buildJob());
  }

  function toggleMethod(methodId: string) {
    setSelectedMethods((prev) =>
      prev.includes(methodId) ? prev.filter((m) => m !== methodId) : [...prev, methodId]
    );
  }

  return (
    <div className="stack">
      <div>
        <h1 className="page-title">Submit a compression job</h1>
        <p className="muted">
          Upload your model artifact and test set to compress, benchmark, and evaluate promotion gates.
        </p>
      </div>

      <form className="card submit-form" onSubmit={onSubmit} noValidate>
        <div className="submit-form__grid">
          {/* ── 1. Model identity & pointers ── */}
          <div className="field field--full">
            <label htmlFor="model-name">
              Model name <span className="field__hint">(optional)</span>
            </label>
            <input
              id="model-name"
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. resnet-50 v1, churn-predictor"
            />
          </div>

          <FileUploadField
            id="model-file"
            label="Model file"
            accept=".pkl,.pickle,.pt,.pth,.safetensors,.onnx,.bin"
            hint="Choose model file (.pt, .pth, .pkl, .onnx, .safetensors)"
            kind="model"
            onUploaded={handleModelUploaded}
          />

          <FileUploadField
            id="test-file"
            label="Test data file"
            accept=".csv,.pt,.bin,.npy"
            hint="Choose test data (.csv, .pt, .npy)"
            kind="test_data"
            onUploaded={handleTestUploaded}
          />

          {/* ── 2. Target column (Show only for tabular models) ── */}
          {isTabular && (
            <div className="field field--full">
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
          )}

          {/* ── 3. Target hardware ── */}
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

          {/* ── 4. Quality Tolerance ── */}
          <div className="field field--full">
            <label>Quality tolerance</label>
            <div className="pill-group">
              <button
                type="button"
                className={`pill-btn ${tolerancePreset === 'strict' ? 'pill-btn--active' : ''}`}
                onClick={() => setTolerancePreset('strict')}
              >
                Strict (0.5%)
                <span className="pill-btn__hint">Minimal accuracy drop</span>
              </button>
              <button
                type="button"
                className={`pill-btn ${tolerancePreset === 'standard' ? 'pill-btn--active' : ''}`}
                onClick={() => setTolerancePreset('standard')}
              >
                Standard (1.0%)
                <span className="pill-btn__hint">Recommended balance</span>
              </button>
              <button
                type="button"
                className={`pill-btn ${tolerancePreset === 'relaxed' ? 'pill-btn--active' : ''}`}
                onClick={() => setTolerancePreset('relaxed')}
              >
                Relaxed (5.0%)
                <span className="pill-btn__hint">Max compression focus</span>
              </button>
              <button
                type="button"
                className={`pill-btn ${tolerancePreset === 'custom' ? 'pill-btn--active' : ''}`}
                onClick={() => setTolerancePreset('custom')}
              >
                Custom %
                <span className="pill-btn__hint">Manual threshold</span>
              </button>
            </div>

            {tolerancePreset === 'custom' && (
              <div style={{ marginTop: 'var(--space-2)' }}>
                <input
                  type="number"
                  className="input"
                  step="0.1"
                  min="0"
                  max="50"
                  value={customTolerancePct}
                  onChange={(e) => setCustomTolerancePct(e.target.value)}
                  placeholder="e.g. 1.5"
                />
              </div>
            )}
          </div>

          {/* ── 5. Compression Strategy ── */}
          <div className="field field--full">
            <label>Compression strategy</label>
            <div className="strategy-grid">
              <div
                className={`strategy-card ${compressionStrategy === 'ai' ? 'strategy-card--active' : ''}`}
                onClick={() => setCompressionStrategy('ai')}
              >
                <div className="strategy-card__head">
                  <span>AI Recommended</span>
                </div>
                <div className="strategy-card__desc">
                  AI determines optimal algorithms for your target hardware. If unguided, it automatically compresses using all available algorithms.
                </div>
              </div>

              <div
                className={`strategy-card ${compressionStrategy === 'all' ? 'strategy-card--active' : ''}`}
                onClick={() => setCompressionStrategy('all')}
              >
                <div className="strategy-card__head">
                  <span>Compress All Algorithms</span>
                </div>
                <div className="strategy-card__desc">
                  Bypass AI recommendations and run every algorithm supported by the target hardware (FP32, INT8 Dynamic, INT8 Static, FP16, TRT).
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── 6. Advanced section (collapsible) ── */}
        <div className="advanced">
          <button
            type="button"
            className="advanced__toggle"
            aria-expanded={showAdvanced}
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? '▾ Advanced Settings' : '▸ Advanced Settings'}
          </button>

          {showAdvanced && (
            <div className="advanced__body">
              <div className="submit-form__grid">

                {/* Model & Task Context inside Advanced Settings */}
                <div className="field field--full" style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <label style={{ fontWeight: 600 }}>
                      Model & Task Context <span className="field__hint">(Optional AI prompt/spec)</span>
                    </label>
                    <button
                      type="button"
                      className="pill-btn pill-btn--active"
                      style={{ padding: '4px 12px', fontSize: '12px' }}
                      onClick={handleGetAiRecommendation}
                      disabled={isRecommending}
                    >
                      {isRecommending ? 'Analyzing with AI…' : 'Get AI Recommendation'}
                    </button>
                  </div>

                  <textarea
                    className="input"
                    rows={2}
                    value={contextText}
                    onChange={(e) => setContextText(e.target.value)}
                    placeholder="Describe your model SLA or accuracy constraints (e.g., 'ResNet-50 for medical image classification. Accuracy drop must stay within 0.5%')"
                    style={{ marginBottom: '10px' }}
                  />

                  <div className="file-drop">
                    <label htmlFor="context-file" className="file-drop__button" style={{ fontSize: '12px' }}>
                      <FileText size={14} style={{ marginRight: '6px' }} /> Upload Context File (.pdf, .docx, .md, .txt)
                    </label>
                    <input
                      id="context-file"
                      type="file"
                      accept=".pdf,.docx,.md,.txt"
                      className="file-drop__input"
                      onChange={handleContextFileChange}
                    />
                    <span className="file-drop__status">
                      {contextFileName ? (
                        <span className="file-drop__ok">✓ Context loaded: {contextFileName}</span>
                      ) : (
                        'Attach spec file or README'
                      )}
                    </span>
                  </div>

                  {recError && (
                    <span className="form-note" style={{ color: 'var(--color-bad)', marginTop: '6px', display: 'block' }}>
                      {recError}
                    </span>
                  )}

                  {aiRecommendation && (
                    <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(0, 0, 0, 0.2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border)' }}>
                      <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '4px', color: '#a5b4fc' }}>
                        AI Recommendation Result
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--color-fg)', lineHeight: 1.4 }}>
                        {aiRecommendation.reasoning}
                      </div>
                      {aiRecommendation.context_summary && (
                        <div style={{ fontSize: '11px', color: 'var(--color-muted)', marginTop: '4px' }}>
                          Summary: {aiRecommendation.context_summary}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Framework Selector inside Advanced */}
                <div className="field">
                  <label htmlFor="framework-select">Framework</label>
                  <select
                    id="framework-select"
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

                {/* Task Selector inside Advanced */}
                <div className="field">
                  <label htmlFor="task-select">Task</label>
                  <select
                    id="task-select"
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
                  <label>
                    Specific compression algorithms{' '}
                    <span className="field__hint">(Select specific methods to override defaults)</span>
                  </label>
                  <div className="method-checkbox-grid">
                    {ALL_COMPRESSORS.map((c) => (
                      <label key={c.id} className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={selectedMethods.includes(c.id)}
                          onChange={() => {
                            toggleMethod(c.id);
                            setCompressionStrategy('custom');
                          }}
                        />
                        <span>{c.label}</span>
                      </label>
                    ))}
                  </div>
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
            {mutation.isPending ? 'Submitting…' : 'Submit compression job'}
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
