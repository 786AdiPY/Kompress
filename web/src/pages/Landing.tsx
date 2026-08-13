// Landing — the marketing front door at "/".
//
// Editorial register: an achromatic palette, Lexend (shared with the console),
// and scroll mechanics that carry the argument rather than decorate it. The five
// chapters run: the problem → the pipeline (scroll-pinned, horizontal) → proof
// from a real run → the gate refusing a bad trade → the platform surface.
//
// Every figure on this page comes from an actual pipeline run (see the fineprint
// in each section); nothing here is illustrative.
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import {
  CountUp,
  Reveal,
  RevealWords,
  useActiveSection,
  useLinePlayback,
  useMagnetic,
  usePageProgress,
  useReducedMotion,
  useScrolled,
  useSectionProgress,
  useTilt3D,
} from '../lib/motion';
import './Landing.css';

// ── the reference run (sklearn MLP fraud classifier, cpu-generic) ────────────
const RUN = {
  nativeKb: 149.3,
  compressedKb: 48.6,
  nativeMs: 0.477,
  compressedMs: 0.272,
  accuracy: 0.8595,
  auc: 0.8534,
  rows: 2000,
  params: 12033,
};

const CHAPTERS = [
  { id: 'problem', label: 'The problem' },
  { id: 'pipeline', label: 'The pipeline' },
  { id: 'proof', label: 'Proof' },
  { id: 'gate', label: 'The gate' },
  { id: 'platform', label: 'Platform' },
];

export default function Landing() {
  // Deep links (/#pipeline) land before React has mounted the sections, so the
  // browser's own hash scroll finds nothing. Re-apply it once painted.
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (!id) return;
    requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ block: 'start' });
    });
  }, []);

  return (
    <div className="lp" id="top">
      <ProgressRail />
      <Nav />
      <ChapterRail />
      <main className="lp-main">
        <Hero />
        <Ticker />
        <Problem />
        <Pipeline />
        <Proof />
        <Gate />
        <Platform />
        <Cta />
      </main>
      <Footer />
    </div>
  );
}

// ── reading progress ─────────────────────────────────────────────────────────
function ProgressRail() {
  const p = usePageProgress();
  return (
    <div className="lp-rail" aria-hidden="true">
      <div className="lp-rail__fill" style={{ transform: `scaleX(${p})` }} />
    </div>
  );
}

// ── nav ──────────────────────────────────────────────────────────────────────
function Nav() {
  const scrolled = useScrolled(24);
  return (
    <header className={`lp-nav ${scrolled ? 'is-scrolled' : ''}`}>
      <a className="lp-nav__brand" href="#top">
        <Mark />
        <span>Kompress</span>
      </a>
      <nav className="lp-nav__links" aria-label="Sections">
        <a href="#problem">Problem</a>
        <a href="#pipeline">Pipeline</a>
        <a href="#proof">Proof</a>
        <a href="#platform">Platform</a>
      </nav>
      <Link to="/dashboard" className="lp-nav__cta">
        Open console
        <Arrow />
      </Link>
    </header>
  );
}

function Mark() {
  return (
    <svg viewBox="0 0 24 24" className="lp-mark" aria-hidden="true">
      <path
        d="M7 4v4H3M17 4v4h4M7 20v-4H3M17 20v-4h4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Arrow() {
  return (
    <svg viewBox="0 0 16 16" className="lp-arrow" aria-hidden="true">
      <path
        d="M3 8h10M9 4l4 4-4 4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ── chapter rail (desktop only, decorative nav) ──────────────────────────────
const CHAPTER_IDS = CHAPTERS.map((c) => c.id);

function ChapterRail() {
  const active = useActiveSection(CHAPTER_IDS);

  return (
    <nav className="lp-chapters" aria-label="Chapters">
      <ol>
        {CHAPTERS.map((c, i) => (
          <li key={c.id} className={active === c.id ? 'is-active' : ''}>
            <a href={`#${c.id}`}>
              <span className="lp-chapters__n">{String(i + 1).padStart(2, '0')}</span>
              <span className="lp-chapters__l">{c.label}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}

// ── hero ─────────────────────────────────────────────────────────────────────
function Hero() {
  const cta = useMagnetic<HTMLAnchorElement>(0.22);

  return (
    <section className="lp-hero">
      <div className="lp-hero__inner">
        <p className="lp-eyebrow">
          <span className="lp-eyebrow__dot" aria-hidden="true" />
          Model compression — automated, gated, reversible
        </p>

        <h1 className="lp-display">
          <RevealWords as="span" className="lp-display__line" text="Smaller models." />
          <RevealWords
            as="span"
            className="lp-display__line lp-display__line--em"
            text="Same answers."
            start={2}
          />
        </h1>

        <Reveal className="lp-hero__lead" delay={4}>
          <p>
            Kompress takes a model you already trained, compresses it every way its
            target hardware allows, and benchmarks each variant against the
            original. Anything that loses more accuracy than you permit is
            blocked — not flagged. You approve what ships.
          </p>
        </Reveal>

        <Reveal className="lp-hero__actions" delay={5}>
          <Link ref={cta} to="/dashboard" className="lp-btn lp-btn--solid">
            Open the console
            <Arrow />
          </Link>
          <a href="#pipeline" className="lp-btn lp-btn--ghost">
            See how it works
          </a>
        </Reveal>

        <Reveal className="lp-spec" delay={6}>
          <SpecItem k="Frameworks" v="XGBoost · LightGBM · sklearn · PyTorch" />
          <SpecItem k="Targets" v="CPU · NVIDIA · ARM NPU · IMX500" />
          <SpecItem k="Gate" v="accuracy · AUC · RMSE" />
          <SpecItem k="Registry" v="MLflow, with rollback" />
        </Reveal>
      </div>

      <Reveal className="lp-hero__figure" delay={4}>
        <HeroFigure />
      </Reveal>

    </section>
  );
}

function SpecItem({ k, v }: { k: string; v: string }) {
  return (
    <div className="lp-spec__item">
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

/** Minimal before/after: two hairline bars, the compressed one in ink so the
 * eye lands on the win.
 *
 * Three nested wrappers on purpose, so the motions compose instead of
 * overwriting one another's `transform`: `.lp-float` runs the idle bob,
 * `.lp-fig3d` takes the cursor tilt, and the figure itself holds the
 * preserve-3d children that parallax at different depths. */
function HeroFigure() {
  const tilt = useTilt3D<HTMLDivElement>(10);
  const ratio = RUN.compressedKb / RUN.nativeKb;

  return (
    <div className="lp-float">
      <div className="lp-fig3d" ref={tilt}>
        <figure className="lp-fig">
          <span className="lp-fig__glare" aria-hidden="true" />

          <figcaption className="lp-fig__head" data-depth="3">
            <span className="lp-mono">fraud_scorer · sklearn</span>
            <span className="lp-tag lp-tag--pass">Gate passed</span>
          </figcaption>

          <div className="lp-fig__row" data-depth="2">
            <span className="lp-fig__k">Original</span>
            <div className="lp-fig__track">
              <div className="lp-fig__bar" style={{ width: '100%' }} />
            </div>
            <span className="lp-fig__v lp-mono">{RUN.nativeKb} KB</span>
          </div>

          <div className="lp-fig__row" data-depth="2">
            <span className="lp-fig__k">Compressed</span>
            <div className="lp-fig__track">
              <div
                className="lp-fig__bar lp-fig__bar--accent"
                style={{ width: `${ratio * 100}%` }}
              />
            </div>
            <span className="lp-fig__v lp-mono">{RUN.compressedKb} KB</span>
          </div>

          <div className="lp-fig__foot" data-depth="4">
            <div>
              <span className="lp-fig__big lp-mono">1.75×</span>
              <span className="lp-fig__lbl">faster</span>
            </div>
            <div>
              <span className="lp-fig__big lp-mono">0.00</span>
              <span className="lp-fig__lbl">accuracy lost</span>
            </div>
          </div>
        </figure>
      </div>
    </div>
  );
}

// ── framework ticker ─────────────────────────────────────────────────────────
const MARQUEE = [
  'XGBoost',
  'LightGBM',
  'scikit-learn',
  'PyTorch',
  'ONNX Runtime',
  'TensorRT',
  'MLflow',
];

function Ticker() {
  return (
    <div className="lp-ticker" aria-hidden="true">
      <div className="lp-ticker__track">
        {[0, 1].map((copy) => (
          <div className="lp-ticker__group" key={copy}>
            {MARQUEE.map((m) => (
              <span key={`${copy}-${m}`} className="lp-ticker__item">
                {m}
                <i />
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── 01 problem ───────────────────────────────────────────────────────────────
function Problem() {
  return (
    <section className="lp-section" id="problem">
      <div className="lp-wrap">
        <SectionLabel n="01" t="The problem" />
        <h2 className="lp-h2">
          <RevealWords text="Compression is easy." as="span" className="lp-h2__l" />
          <RevealWords
            text="Trusting it is the hard part."
            as="span"
            className="lp-h2__l lp-h2__l--em"
            start={3}
          />
        </h2>

        <div className="lp-cols">
          <Reveal as="p" delay={1}>
            Quantize a model and it gets smaller. It also, quietly, gets worse — a
            fraction of a point of AUC here, a shifted decision boundary there. The
            damage rarely shows up in a smoke test. It shows up in production, weeks
            later, in exactly the cases that mattered most.
          </Reveal>
          <Reveal as="p" delay={2}>
            So teams do the safe thing: they don't compress. They pay for the larger
            instance, accept the slower response, and leave the win on the table —
            because nobody wants to be the person who shipped a cheaper model that
            turned out to be a worse one.
          </Reveal>
        </div>

        <Reveal className="lp-quote" delay={3}>
          <blockquote>
            The blocker was never the compression. It was the absence of proof.
          </blockquote>
        </Reveal>
      </div>
    </section>
  );
}

function SectionLabel({ n, t }: { n: string; t: string }) {
  return (
    <Reveal className="lp-label">
      <span className="lp-mono">{n}</span>
      <i aria-hidden="true" />
      <span>{t}</span>
    </Reveal>
  );
}

// ── 02 pipeline — scroll-pinned horizontal ───────────────────────────────────
const STAGES = [
  {
    k: 'Ingest',
    t: 'Point at a model.',
    d: 'Hand over a trained artifact and the test set you already have. No retraining, no training data, no code changes to the model itself.',
    m: 'pkl · pt',
  },
  {
    k: 'Export',
    t: 'Normalize to one graph.',
    d: 'An adapter turns each framework into a single portable ONNX graph. Every stage after this one — compressors, benchmark, gate — works on that one representation, so adding a framework changes nothing downstream.',
    m: 'one graph, any source',
  },
  {
    k: 'Compress',
    t: 'Try everything the hardware allows.',
    d: 'The deployment target selects the techniques: dynamic and static INT8 for CPU, TensorRT INT8 for NVIDIA, static INT8 for NPU-class devices.',
    m: 'hardware-derived',
  },
  {
    k: 'Benchmark',
    t: 'Measure, never assume.',
    d: 'Each variant runs the same test set on the same machine — warmed up, repeated, and scored on latency, size and the full task metrics.',
    m: `${RUN.rows.toLocaleString()} rows`,
  },
  {
    k: 'Gate',
    t: 'Refuse the bad trades.',
    d: 'Accuracy, AUC and RMSE deltas are compared against thresholds you set. A variant that gives up more than you allowed cannot be promoted. Not a warning — a block.',
    m: 'Δacc ≤ 0.01',
  },
  {
    k: 'Register',
    t: 'Promote only on consent.',
    d: 'The winner lands in MLflow with its full provenance — base model hash, chosen variant, every delta, the gate report. A human approves before Production.',
    m: 'MLflow registry',
  },
  {
    k: 'Deploy',
    t: 'Export to where it runs.',
    d: 'Take the compressed artifact straight from the run, or export it for the device that will serve it. Roll back to any previous version in a single action.',
    m: 'onnx · tensorrt',
  },
];

function Pipeline() {
  const reduced = useReducedMotion();
  const [progressRef, p] = useSectionProgress<HTMLElement>('pin');
  const vpRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const [overflow, setOverflow] = useState(0);

  // Horizontal travel = how much wider the track is than its viewport. The
  // section is then made that much taller, so 1px of scroll = 1px of pan.
  useLayoutEffect(() => {
    if (reduced) {
      setOverflow(0);
      return;
    }
    const measure = () => {
      const vp = vpRef.current;
      const track = trackRef.current;
      if (!vp || !track) return;
      setOverflow(Math.max(0, track.scrollWidth - vp.clientWidth));
    };
    measure();
    window.addEventListener('resize', measure);
    const t = window.setTimeout(measure, 300); // after webfonts settle
    return () => {
      window.removeEventListener('resize', measure);
      window.clearTimeout(t);
    };
  }, [reduced]);

  const activeIndex = Math.min(
    STAGES.length - 1,
    Math.round(p * (STAGES.length - 1)),
  );

  return (
    <section
      className={`lp-pin ${reduced ? 'is-static' : ''}`}
      id="pipeline"
      ref={progressRef}
      style={overflow ? { height: `calc(100vh + ${overflow}px)` } : undefined}
    >
      <div className="lp-pin__stage">
        <div className="lp-wrap lp-pin__head">
          <SectionLabel n="02" t="The pipeline" />
          <h2 className="lp-h2 lp-h2--tight">
            Seven stages, every one of them auditable.
          </h2>
          <div className="lp-pin__meter" aria-hidden="true">
            <div className="lp-pin__meter-fill" style={{ transform: `scaleX(${p})` }} />
          </div>
        </div>

        <div className="lp-pin__vp" ref={vpRef}>
          <div
            className="lp-pin__track"
            ref={trackRef}
            style={
              reduced ? undefined : { transform: `translate3d(${-p * overflow}px,0,0)` }
            }
          >
            {STAGES.map((s, i) => (
              <article
                className={`lp-stage ${i === activeIndex ? 'is-active' : ''}`}
                key={s.k}
              >
                <header>
                  <span className="lp-stage__n lp-mono">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span className="lp-stage__k lp-mono">{s.k}</span>
                </header>
                <h3>{s.t}</h3>
                <p>{s.d}</p>
                <footer className="lp-mono">{s.m}</footer>
              </article>
            ))}
          </div>
        </div>

        <div className="lp-wrap lp-pin__dots" aria-hidden="true">
          {STAGES.map((s, i) => (
            <span key={s.k} className={i <= activeIndex ? 'is-on' : ''} />
          ))}
        </div>
      </div>
    </section>
  );
}

// ── 03 proof ─────────────────────────────────────────────────────────────────
function Proof() {
  return (
    <section className="lp-section lp-section--ink" id="proof">
      <div className="lp-wrap">
        <SectionLabel n="03" t="Proof" />
        <h2 className="lp-h2">
          <RevealWords text="Numbers from a run," as="span" className="lp-h2__l" />
          <RevealWords
            text="not from a deck."
            as="span"
            className="lp-h2__l lp-h2__l--em"
            start={4}
          />
        </h2>

        <div className="lp-metrics">
          <Metric v={<CountUp to={67.45} decimals={2} suffix="%" />} l="smaller on disk" s="149.3 KB → 48.6 KB" />
          <Metric v={<CountUp to={1.75} decimals={2} suffix="×" />} l="faster inference" s="0.477 ms → 0.272 ms" />
          <Metric v={<CountUp to={0} decimals={2} />} l="accuracy lost" s="0.8595 → 0.8595" />
          <Metric v={<CountUp to={0} decimals={2} />} l="AUC lost" s="0.8534 → 0.8534" />
        </div>

        <Compressor />

        <Reveal as="p" className="lp-fineprint" delay={2}>
          A {RUN.params.toLocaleString()}-parameter scikit-learn MLP (fraud
          classifier) compressed for <span className="lp-mono">cpu-generic</span> and
          scored on a held-out {RUN.rows.toLocaleString()}-row test set. Accuracy and
          AUC are identical before and after, to four decimal places.
        </Reveal>
      </div>
    </section>
  );
}

function Metric({ v, l, s }: { v: React.ReactNode; l: string; s: string }) {
  return (
    <Reveal className="lp-metric">
      <span className="lp-metric__v lp-mono">{v}</span>
      <span className="lp-metric__l">{l}</span>
      <span className="lp-metric__s lp-mono">{s}</span>
    </Reveal>
  );
}

/** Drag to interpolate the model between its original and compressed state. */
function Compressor() {
  const [t, setT] = useState(0);
  const lerp = (a: number, b: number) => a + (b - a) * t;
  const kb = lerp(RUN.nativeKb, RUN.compressedKb);
  const ms = lerp(RUN.nativeMs, RUN.compressedMs);

  return (
    <Reveal className="lp-compressor" delay={1}>
      <div className="lp-compressor__head">
        <label htmlFor="lp-squeeze" className="lp-mono">
          Drag to compress
        </label>
        <span className="lp-compressor__state lp-mono">
          {t < 0.02 ? 'original' : t > 0.98 ? 'compressed' : 'compressing…'}
        </span>
      </div>

      <div className="lp-compressor__bar">
        <div
          className="lp-compressor__fill"
          style={{ width: `${(kb / RUN.nativeKb) * 100}%` }}
        />
      </div>

      <input
        id="lp-squeeze"
        className="lp-range"
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={t}
        onChange={(e) => setT(Number(e.target.value))}
        aria-valuetext={`${kb.toFixed(1)} kilobytes, ${ms.toFixed(3)} milliseconds`}
      />

      <div className="lp-compressor__read">
        <span className="lp-mono">{kb.toFixed(1)} KB</span>
        <span className="lp-mono">{ms.toFixed(3)} ms</span>
      </div>
    </Reveal>
  );
}

// ── 04 the gate ──────────────────────────────────────────────────────────────
const VARIANTS = [
  { n: 'onnx_fp32', kb: 48.6, x: '1.75×', acc: '0.0000', auc: '0.0000', pass: true },
  { n: 'onnx_int8_dynamic', kb: 17.4, x: '2.00×', acc: '−0.0220', auc: '−0.0521', pass: false },
  { n: 'onnx_int8_static', kb: 21.0, x: '1.34×', acc: '−0.0295', auc: '−0.0628', pass: false },
];

const TERMINAL = [
  { t: 'cmd', v: 'kompress run --model fraud_scorer.pkl --target cpu-generic' },
  { t: 'out', v: 'export     onnx_fp32            48.6 KB' },
  { t: 'out', v: 'compress   onnx_int8_dynamic    17.4 KB' },
  { t: 'out', v: 'compress   onnx_int8_static     21.0 KB' },
  { t: 'out', v: 'benchmark  2,000 rows × 12 features' },
  { t: 'bad', v: 'gate       onnx_int8_dynamic    BLOCKED   Δacc −0.0220 > 0.0100' },
  { t: 'bad', v: 'gate       onnx_int8_static     BLOCKED   Δacc −0.0295 > 0.0100' },
  { t: 'ok', v: 'gate       onnx_fp32             PASS      Δacc  0.0000' },
  { t: 'ok', v: 'registered fraud_scorer → pending approval' },
];

function Gate() {
  const [ref, shown] = useLinePlayback(TERMINAL.length, 240);

  return (
    <section className="lp-section" id="gate">
      <div className="lp-wrap">
        <SectionLabel n="04" t="The gate" />
        <h2 className="lp-h2">
          <RevealWords text="The most valuable thing" as="span" className="lp-h2__l" />
          <RevealWords
            text="it does is say no."
            as="span"
            className="lp-h2__l lp-h2__l--em"
            start={4}
          />
        </h2>

        <div className="lp-gate">
          <div className="lp-gate__copy">
            <Reveal as="p" delay={1}>
              In the very same run, INT8 quantization produced a model{' '}
              <strong>88% smaller and twice as fast</strong>. It also gave up 2.2
              points of accuracy and 5.2 of AUC.
            </Reveal>
            <Reveal as="p" delay={2}>
              Both INT8 variants were blocked, and the honest win shipped instead.
              That decision is the product. A compression tool that only ever says
              yes is just a slower way to lose accuracy.
            </Reveal>

            <Reveal className="lp-table" delay={3}>
              <table>
                <caption className="lp-visually-hidden">
                  Variants from the reference run and their gate outcome
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Variant</th>
                    <th scope="col">Size</th>
                    <th scope="col">Speed</th>
                    <th scope="col">Δ acc</th>
                    <th scope="col">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {VARIANTS.map((v) => (
                    <tr key={v.n} className={v.pass ? 'is-pass' : ''}>
                      <td className="lp-mono">{v.n}</td>
                      <td className="lp-mono">{v.kb.toFixed(1)} KB</td>
                      <td className="lp-mono">{v.x}</td>
                      <td className="lp-mono">{v.acc}</td>
                      <td>
                        <span className={`lp-tag ${v.pass ? 'lp-tag--pass' : 'lp-tag--block'}`}>
                          {v.pass ? 'Shipped' : 'Blocked'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Reveal>
          </div>

          <div className="lp-term" ref={ref}>
            <div className="lp-term__bar" aria-hidden="true">
              <i />
              <i />
              <i />
              <span className="lp-mono">run · fraud_scorer</span>
            </div>
            <pre className="lp-term__body" aria-label="Example pipeline run output">
              {TERMINAL.slice(0, shown).map((l, i) => (
                <div className={`lp-term__l lp-term__l--${l.t}`} key={i}>
                  {l.t === 'cmd' && <span className="lp-term__p">$</span>}
                  {l.v}
                </div>
              ))}
              {shown < TERMINAL.length && <span className="lp-term__caret" />}
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

// ── 05 platform — sticky stacked cards ───────────────────────────────────────
const CAPS = [
  {
    k: 'One contract, any framework',
    d: 'A job is a model pointer, a test set, and the hardware it will run on. Adapters cover XGBoost, LightGBM, scikit-learn and PyTorch behind that single shape — adding a framework does not change the interface.',
  },
  {
    k: 'Hardware decides the techniques',
    d: 'Target hardware is not a label on the benchmark; it selects the compressors. CPU targets get ONNX Runtime INT8, NVIDIA gets a TensorRT engine, NPU classes get static INT8 — and anything unavailable is skipped rather than failing the run.',
  },
  {
    k: 'A gate you configure',
    d: 'Set the maximum accuracy, AUC or RMSE you are willing to trade. The thresholds are per-job, enforced in the pipeline rather than in review, and recorded with the result.',
  },
  {
    k: 'Provenance, not vibes',
    d: 'Every run stores the base model hash, the variant chosen, all deltas and the gate report in MLflow — so six months later you can answer exactly what shipped and why it was allowed to.',
  },
  {
    k: 'Rollback as a first-class action',
    d: 'Production is a pointer, and it moves both ways. Any previous version can be restored from the console without a redeploy or a rebuild.',
  },
];

function Platform() {
  return (
    <section className="lp-section" id="platform">
      <div className="lp-wrap">
        <SectionLabel n="05" t="Platform" />
        <h2 className="lp-h2 lp-h2--tight">Built to be handed to a team.</h2>
      </div>

      <div className="lp-wrap lp-stack">
        {CAPS.map((c, i) => (
          <article
            className="lp-stack__card"
            key={c.k}
            style={{ ['--i' as string]: i }}
          >
            <span className="lp-stack__n lp-mono">{String(i + 1).padStart(2, '0')}</span>
            <div>
              <h3>{c.k}</h3>
              <p>{c.d}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

// ── cta ──────────────────────────────────────────────────────────────────────
function Cta() {
  const ref = useMagnetic<HTMLAnchorElement>(0.2);
  return (
    <section className="lp-cta">
      <div className="lp-wrap">
        <h2 className="lp-display lp-display--cta">
          <RevealWords as="span" className="lp-display__line" text="Compress something." />
        </h2>
        <Reveal className="lp-cta__actions" delay={2}>
          <Link ref={ref} to="/dashboard" className="lp-btn lp-btn--solid lp-btn--lg">
            Open the console
            <Arrow />
          </Link>
          <Link to="/submit" className="lp-btn lp-btn--ghost lp-btn--lg">
            Submit a job
          </Link>
        </Reveal>
      </div>
    </section>
  );
}

// ── footer ───────────────────────────────────────────────────────────────────
function Footer() {
  return (
    <footer className="lp-footer">
      <div className="lp-wrap lp-footer__in">
        <a className="lp-nav__brand" href="#top">
          <Mark />
          <span>Kompress</span>
        </a>
        <nav aria-label="Footer">
          <Link to="/dashboard">Console</Link>
          <Link to="/submit">Submit</Link>
          <Link to="/deployments">Deployments</Link>
          <a href="#pipeline">Pipeline</a>
        </nav>
        <p className="lp-mono">Compress · Benchmark · Gate · Register · Deploy</p>
      </div>
    </footer>
  );
}
