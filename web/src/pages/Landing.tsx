// Marketing landing page (route "/"). Its own full-bleed chrome — distinct from
// the product app shell. Editorial, scroll-reactive, and honest about results.
import { Link } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  Boxes,
  ClipboardCheck,
  Cpu,
  Gauge,
  GitBranch,
  Layers,
  Rocket,
  ShieldCheck,
  Shrink,
  Sparkles,
  Upload,
  type LucideIcon,
} from 'lucide-react';

import { CountUp, Reveal, useScrolled } from '../lib/motion';
import './Landing.css';

// ── content model ────────────────────────────────────────────────────────────
const PIPELINE: { icon: LucideIcon; title: string; blurb: string }[] = [
  { icon: Boxes, title: 'Train', blurb: 'Train a baseline or bring your own model' },
  { icon: Shrink, title: 'Compress', blurb: 'Every method that applies to the model' },
  { icon: Gauge, title: 'Benchmark', blurb: 'Latency, size & quality on your hardware' },
  { icon: ShieldCheck, title: 'Gate', blurb: 'Block anything that loses accuracy' },
  { icon: GitBranch, title: 'Register', blurb: 'Version the winner in the registry' },
  { icon: Rocket, title: 'Deploy', blurb: 'Promote to production behind a health check' },
  { icon: Activity, title: 'Monitor', blurb: 'Watch live traffic for feature drift' },
];

const FEATURES: { icon: LucideIcon; title: string; body: string }[] = [
  {
    icon: Layers,
    title: 'Framework-agnostic',
    body: 'XGBoost, LightGBM, scikit-learn and PyTorch through a single pipeline — one small adapter per framework, the rest is shared.',
  },
  {
    icon: Shrink,
    title: 'Every compression method',
    body: 'ONNX FP32 export, INT8 dynamic & static quantization, and TensorRT INT8 on GPU. Each variant is built, measured, and the best one wins.',
  },
  {
    icon: ShieldCheck,
    title: 'An accuracy gate that means it',
    body: 'A variant that drops accuracy, AUC, or RMSE past your threshold is auto-rejected. Fast-but-degraded models never reach production.',
  },
  {
    icon: Gauge,
    title: 'Benchmarked, not guessed',
    body: 'Warm-run latency, model size, and task metrics measured on your target hardware — every variant reported with its speedup vs native.',
  },
  {
    icon: GitBranch,
    title: 'Registry & one-click rollback',
    body: 'Winners are registered and promoted to Production with full lineage. Re-point any prior version back to Production in a click.',
  },
  {
    icon: Activity,
    title: 'Drift monitoring built in',
    body: 'Population-stability checks compare live features against the training baseline and flag drift before it quietly costs you accuracy.',
  },
];

const STEPS: { icon: LucideIcon; title: string; body: string }[] = [
  {
    icon: Upload,
    title: 'Submit a pointer',
    body: 'Point Kompress at a trained model and a test set — an S3 or MLflow URI. Pointers only; your files are never uploaded.',
  },
  {
    icon: Cpu,
    title: 'Compress & benchmark',
    body: 'The pipeline exports, quantizes, and benchmarks every applicable variant on the hardware you chose.',
  },
  {
    icon: ClipboardCheck,
    title: 'Review the plan',
    body: 'See size, latency and quality deltas for each variant next to the promotion gate — then approve or reject.',
  },
  {
    icon: Rocket,
    title: 'Deploy & watch',
    body: 'Approved models register to Production and deploy. Drift monitoring takes it from there.',
  },
];

// Real figures from a reference run (sklearn regressor) — see note under the grid.
const LATENCY = { native: 31.4, compressed: 4.2 }; // ms
const SIZE = { native: 52.6, compressed: 28.8 }; // MB

export default function Landing() {
  const scrolled = useScrolled(24);

  return (
    <div className="lp" id="top">
      {/* ── nav ─────────────────────────────────────────────────────────── */}
      <header className={`lp-nav ${scrolled ? 'is-scrolled' : ''}`}>
        <div className="lp-nav__inner">
          <Link to="/" className="lp-brand" aria-label="Kompress home">
            <span className="lp-brand__mark" aria-hidden="true">
              <Shrink size={18} strokeWidth={2.4} />
            </span>
            <span className="lp-brand__name">Kompress</span>
          </Link>

          <nav className="lp-nav__links" aria-label="Sections">
            <a href="#pipeline">Pipeline</a>
            <a href="#features">Platform</a>
            <a href="#results">Results</a>
            <a href="#how">How it works</a>
          </nav>

          <Link to="/dashboard" className="lp-btn lp-btn--primary lp-nav__cta">
            Go to Dashboard
            <ArrowRight size={16} aria-hidden="true" />
          </Link>
        </div>
      </header>

      {/* ── hero ────────────────────────────────────────────────────────── */}
      <section className="lp-hero">
        <div className="lp-hero__aurora" aria-hidden="true">
          <span className="blob blob--1" />
          <span className="blob blob--2" />
          <span className="blob blob--3" />
          <span className="lp-grid" />
        </div>

        <div className="lp-hero__inner">
          <div className="lp-hero__copy">
            <p className="lp-eyebrow">
              <Sparkles size={14} aria-hidden="true" />
              Model compression, automated end to end
            </p>
            <h1 className="lp-hero__title">
              Ship models that are <span className="grad">smaller and faster</span>{' '}
              — without losing accuracy.
            </h1>
            <p className="lp-hero__sub">
              Kompress takes any trained model, compresses it every way that applies,
              benchmarks the variants on your hardware, and blocks anything that drops
              accuracy. Then it registers the winner and ships it — reproducibly,
              with a human in the loop.
            </p>
            <div className="lp-hero__cta">
              <Link to="/dashboard" className="lp-btn lp-btn--primary lp-btn--lg">
                Go to Dashboard
                <ArrowRight size={18} aria-hidden="true" />
              </Link>
              <a href="#how" className="lp-btn lp-btn--ghost lp-btn--lg">
                See how it works
              </a>
            </div>
            <p className="lp-hero__compat">
              Works with
              <b> XGBoost</b> · <b>LightGBM</b> · <b>scikit-learn</b> · <b>PyTorch</b>
            </p>
          </div>

          {/* floating benchmark card — honest reference numbers */}
          <div className="lp-hero__card-wrap">
            <div className="lp-card lp-card--float">
              <div className="lp-card__head">
                <span className="lp-card__title">Benchmark · house_price</span>
                <span className="lp-chip lp-chip--good">
                  <ShieldCheck size={13} aria-hidden="true" /> Gate passed
                </span>
              </div>

              <BarRow label="Native" value={LATENCY.native} max={LATENCY.native} unit="ms" tone="muted" />
              <BarRow
                label="Compressed"
                value={LATENCY.compressed}
                max={LATENCY.native}
                unit="ms"
                tone="accent"
              />

              <div className="lp-card__foot">
                <div>
                  <span className="lp-card__big">7.5×</span>
                  <span className="lp-card__cap">faster inference</span>
                </div>
                <div>
                  <span className="lp-card__big">−45%</span>
                  <span className="lp-card__cap">smaller on disk</span>
                </div>
                <div>
                  <span className="lp-card__big">0.0%</span>
                  <span className="lp-card__cap">accuracy lost</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <a href="#problem" className="lp-scroll-hint" aria-label="Scroll to content">
          <span />
        </a>
      </section>

      {/* ── problem ─────────────────────────────────────────────────────── */}
      <section className="lp-section" id="problem">
        <Reveal className="lp-section__head" as="header">
          <p className="lp-kicker">The problem</p>
          <h2 className="lp-h2">
            Big models win offline. Then they have to run in production.
          </h2>
          <p className="lp-lede">
            Accuracy and serving cost pull in opposite directions — and hand-tuned
            compression trades away quality you can't easily see.
          </p>
        </Reveal>

        <div className="lp-tiles">
          {[
            {
              k: 'Latency & cost',
              v: 'Large models are slow and expensive to serve at scale — every millisecond and megabyte is a bill.',
            },
            {
              k: 'Silent accuracy loss',
              v: 'Quantize by hand and you might ship a model that is quietly worse. Nobody notices until it matters.',
            },
            {
              k: 'It doesn’t generalize',
              v: 'Every framework and target has its own toolchain. What worked for one model rarely transfers to the next.',
            },
          ].map((t, i) => (
            <Reveal className="lp-tile" key={t.k} delay={i}>
              <span className="lp-tile__num">{String(i + 1).padStart(2, '0')}</span>
              <h3>{t.k}</h3>
              <p>{t.v}</p>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ── pipeline ────────────────────────────────────────────────────── */}
      <section className="lp-section lp-section--tint" id="pipeline">
        <Reveal className="lp-section__head" as="header">
          <p className="lp-kicker">One pipeline, seven stages</p>
          <h2 className="lp-h2">From a trained model to a monitored deployment.</h2>
          <p className="lp-lede">
            Each stage hands a signed artifact to the next. Nothing ships that
            hasn't been measured and gated.
          </p>
        </Reveal>

        <div className="lp-flow">
          {PIPELINE.map((stage, i) => (
            <Reveal className="lp-flow__node" key={stage.title} delay={i}>
              <span className="lp-flow__dot">
                <stage.icon size={20} aria-hidden="true" />
              </span>
              <span className="lp-flow__idx">Step {i + 1}</span>
              <h3>{stage.title}</h3>
              <p>{stage.blurb}</p>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ── features ────────────────────────────────────────────────────── */}
      <section className="lp-section" id="features">
        <Reveal className="lp-section__head" as="header">
          <p className="lp-kicker">The platform</p>
          <h2 className="lp-h2">Everything the loop needs — nothing you have to wire up.</h2>
        </Reveal>

        <div className="lp-features">
          {FEATURES.map((f, i) => (
            <Reveal className="lp-feature" key={f.title} delay={i % 3}>
              <span className="lp-feature__icon">
                <f.icon size={22} aria-hidden="true" />
              </span>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ── results ─────────────────────────────────────────────────────── */}
      <section className="lp-section lp-section--dark" id="results">
        <Reveal className="lp-section__head" as="header">
          <p className="lp-kicker lp-kicker--on-dark">Proven on a real run</p>
          <h2 className="lp-h2">Smaller and faster — with the accuracy gate green.</h2>
        </Reveal>

        <div className="lp-stats">
          {[
            { to: 7.5, dec: 1, suffix: '×', cap: 'peak inference speedup' },
            { to: 45, dec: 0, suffix: '%', cap: 'smaller on disk' },
            { to: 0.0, dec: 1, suffix: '%', cap: 'accuracy lost' },
            { to: 4, dec: 0, suffix: '', cap: 'frameworks supported' },
          ].map((s, i) => (
            <Reveal className="lp-stat" key={s.cap} delay={i}>
              <span className="lp-stat__val">
                <CountUp to={s.to} decimals={s.dec} suffix={s.suffix} />
              </span>
              <span className="lp-stat__cap">{s.cap}</span>
            </Reveal>
          ))}
        </div>

        <Reveal className="lp-compare">
          <div className="lp-compare__col">
            <span className="lp-compare__title">Inference latency</span>
            <CompareBar label="Native" value={LATENCY.native} max={LATENCY.native} unit=" ms" tone="muted" />
            <CompareBar label="Compressed" value={LATENCY.compressed} max={LATENCY.native} unit=" ms" tone="accent" />
          </div>
          <div className="lp-compare__col">
            <span className="lp-compare__title">Model size</span>
            <CompareBar label="Native" value={SIZE.native} max={SIZE.native} unit=" MB" tone="muted" />
            <CompareBar label="Compressed" value={SIZE.compressed} max={SIZE.native} unit=" MB" tone="accent" />
          </div>
        </Reveal>

        <p className="lp-fineprint">
          Figures from a reference run on a scikit-learn regressor; gains vary by
          model, framework, and target hardware. Compression is never promoted if
          it fails the accuracy gate.
        </p>
      </section>

      {/* ── how it works ────────────────────────────────────────────────── */}
      <section className="lp-section" id="how">
        <Reveal className="lp-section__head" as="header">
          <p className="lp-kicker">How it works</p>
          <h2 className="lp-h2">Four steps from “trained” to “in production.”</h2>
        </Reveal>

        <div className="lp-steps">
          {STEPS.map((s, i) => (
            <Reveal className="lp-step" key={s.title} delay={i}>
              <div className="lp-step__rail">
                <span className="lp-step__icon">
                  <s.icon size={20} aria-hidden="true" />
                </span>
                {i < STEPS.length - 1 && <span className="lp-step__line" />}
              </div>
              <div className="lp-step__body">
                <span className="lp-step__num">Step {i + 1}</span>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ── CTA band ────────────────────────────────────────────────────── */}
      <section className="lp-cta">
        <Reveal className="lp-cta__inner">
          <h2>Compress your first model in minutes.</h2>
          <p>
            Open the dashboard, submit a job, and watch it get smaller, faster, and
            gated — with the winner one click from production.
          </p>
          <div className="lp-cta__actions">
            <Link to="/dashboard" className="lp-btn lp-btn--primary lp-btn--lg">
              Go to Dashboard
              <ArrowRight size={18} aria-hidden="true" />
            </Link>
            <Link to="/submit" className="lp-btn lp-btn--ghost lp-btn--lg">
              Submit a job
            </Link>
          </div>
        </Reveal>
      </section>

      {/* ── footer ──────────────────────────────────────────────────────── */}
      <footer className="lp-footer">
        <div className="lp-footer__inner">
          <div className="lp-footer__brand">
            <span className="lp-brand__mark" aria-hidden="true">
              <Shrink size={16} strokeWidth={2.4} />
            </span>
            <div>
              <span className="lp-brand__name">Kompress</span>
              <p>Automated model compression, gated on accuracy.</p>
            </div>
          </div>
          <nav className="lp-footer__nav" aria-label="Footer">
            <Link to="/dashboard">Dashboard</Link>
            <Link to="/submit">Submit</Link>
            <Link to="/deployments">Deployments</Link>
            <a href="#pipeline">Pipeline</a>
            <a href="#results">Results</a>
          </nav>
        </div>
        <div className="lp-footer__base">
          <span>Compress · Benchmark · Gate · Deploy · Monitor</span>
        </div>
      </footer>
    </div>
  );
}

// ── small bar used inside the hero card ──────────────────────────────────────
function BarRow({
  label,
  value,
  max,
  unit,
  tone,
}: {
  label: string;
  value: number;
  max: number;
  unit: string;
  tone: 'accent' | 'muted';
}) {
  const pct = Math.max(4, (value / max) * 100);
  return (
    <div className="lp-bar">
      <span className="lp-bar__label">{label}</span>
      <span className="lp-bar__track">
        <span className={`lp-bar__fill lp-bar__fill--${tone}`} style={{ width: `${pct}%` }} />
      </span>
      <span className="lp-bar__val">
        {value}
        {unit}
      </span>
    </div>
  );
}

// ── larger comparison bar for the results section (animates width on reveal) ──
function CompareBar({
  label,
  value,
  max,
  unit,
  tone,
}: {
  label: string;
  value: number;
  max: number;
  unit: string;
  tone: 'accent' | 'muted';
}) {
  const pct = Math.max(5, (value / max) * 100);
  return (
    <div className="lp-cbar">
      <span className="lp-cbar__label">{label}</span>
      <span className="lp-cbar__track">
        <span
          className={`lp-cbar__fill lp-cbar__fill--${tone}`}
          style={{ ['--w' as string]: `${pct}%` }}
        >
          <span className="lp-cbar__val">
            {value}
            {unit}
          </span>
        </span>
      </span>
    </div>
  );
}
