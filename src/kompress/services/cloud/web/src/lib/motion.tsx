// Small, dependency-free scroll-motion primitives for the landing page.
// All respect prefers-reduced-motion: when the user opts out, reveals show
// immediately, counters jump to target, and scroll-pinned sections fall back
// to plain stacked layout.
import {
  createElement,
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ElementType,
  type ReactNode,
} from 'react';

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Live-tracking version of the media query, so layout (not just animation)
 * can branch on it and re-render if the user flips the OS setting. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(prefersReducedMotion);
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (!mq) return;
    const onChange = () => setReduced(mq.matches);
    onChange();
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

/** Returns [refCallback, inView] — inView latches true the first time the
 * element crosses the viewport threshold, then stops observing. Uses a
 * callback ref so it is agnostic to React's ref-nullability typings. */
export function useInView<T extends Element = HTMLDivElement>(
  options: IntersectionObserverInit = { threshold: 0.2, rootMargin: '0px 0px -8% 0px' },
): [(node: T | null) => void, boolean] {
  const [inView, setInView] = useState(false);
  const seen = useRef(false);
  const observer = useRef<IntersectionObserver | null>(null);
  const optionsRef = useRef(options);

  const setRef = useCallback((node: T | null) => {
    observer.current?.disconnect();
    if (seen.current || !node) return;

    if (prefersReducedMotion() || typeof IntersectionObserver === 'undefined') {
      seen.current = true;
      setInView(true);
      return;
    }

    observer.current = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          seen.current = true;
          setInView(true);
          observer.current?.disconnect();
          break;
        }
      }
    }, optionsRef.current);
    observer.current.observe(node);
  }, []);

  return [setRef, inView];
}

/** True once the page has scrolled past `threshold` px. Drives the landing
 * header's transparent → solid transition. */
export function useScrolled(threshold = 8): boolean {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > threshold);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [threshold]);
  return scrolled;
}

// ── shared rAF-throttled scroll subscription ────────────────────────────────
// One listener for the whole page: every scroll-linked hook reads on the same
// frame, so N components cost one layout pass rather than N.
type ScrollFn = () => void;
const subscribers = new Set<ScrollFn>();
let ticking = false;
let listening = false;

function flush() {
  ticking = false;
  for (const fn of subscribers) fn();
}

function onGlobalScroll() {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(flush);
}

function subscribe(fn: ScrollFn): () => void {
  subscribers.add(fn);
  if (!listening) {
    window.addEventListener('scroll', onGlobalScroll, { passive: true });
    window.addEventListener('resize', onGlobalScroll, { passive: true });
    listening = true;
  }
  fn();
  return () => {
    subscribers.delete(fn);
    if (subscribers.size === 0 && listening) {
      window.removeEventListener('scroll', onGlobalScroll);
      window.removeEventListener('resize', onGlobalScroll);
      listening = false;
    }
  };
}

/** 0 → 1 progress of the whole document. Drives the top reading rail. */
export function usePageProgress(): number {
  const [p, setP] = useState(0);
  useEffect(
    () =>
      subscribe(() => {
        const max = document.documentElement.scrollHeight - window.innerHeight;
        setP(max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0);
      }),
    [],
  );
  return p;
}

/** 0 → 1 progress of `ref` travelling through the viewport: 0 when its top
 * hits the viewport bottom, 1 when its bottom leaves the viewport top.
 * `mode: 'pin'` measures a tall section against its own scrollable length,
 * which is what a sticky/pinned child needs. */
export function useSectionProgress<T extends HTMLElement>(
  mode: 'through' | 'pin' = 'through',
): [(node: T | null) => void, number] {
  const [p, setP] = useState(0);
  const node = useRef<T | null>(null);
  const setRef = useCallback((n: T | null) => {
    node.current = n;
  }, []);

  useEffect(
    () =>
      subscribe(() => {
        const el = node.current;
        if (!el) return;
        const r = el.getBoundingClientRect();
        const vh = window.innerHeight;
        const span = mode === 'pin' ? r.height - vh : r.height + vh;
        if (span <= 0) return setP(0);
        const travelled = mode === 'pin' ? -r.top : vh - r.top;
        setP(Math.min(1, Math.max(0, travelled / span)));
      }),
    [mode],
  );

  return [setRef, p];
}

/** Which of `ids` is the current chapter. Deliberately scroll-position based
 * rather than IntersectionObserver: one section (the pinned pipeline) is many
 * viewports tall and would otherwise stay "active" across its whole length. */
export function useActiveSection(ids: string[]): string {
  const [active, setActive] = useState('');
  useEffect(() => {
    const key = ids.join('|');
    return subscribe(() => {
      const line = window.scrollY + window.innerHeight * 0.4;
      let current = '';
      for (const id of key.split('|')) {
        const el = document.getElementById(id);
        if (!el) continue;
        if (el.getBoundingClientRect().top + window.scrollY <= line) current = id;
      }
      setActive(current);
    });
  }, [ids]);
  return active;
}

type RevealProps = {
  children: ReactNode;
  as?: ElementType;
  className?: string;
  /** stagger index — multiplies the base per-item delay */
  delay?: number;
  style?: CSSProperties;
};

/** Fades + lifts its children into view on first scroll-intersection.
 * `delay` staggers siblings (each unit ≈ 70ms via the CSS var). */
export function Reveal({
  children,
  as = 'div',
  className = '',
  delay = 0,
  style,
}: RevealProps) {
  const [setRef, inView] = useInView<HTMLElement>();
  return createElement(
    as,
    {
      ref: setRef,
      className: `reveal ${inView ? 'is-visible' : ''} ${className}`.trim(),
      style: { ['--rvl-i' as string]: delay, ...style },
    },
    children,
  );
}

/** Editorial line-by-line reveal: each child word/line lifts in sequence.
 * Used for the hero and section leads. */
export function RevealWords({
  text,
  as = 'span',
  className = '',
  start = 0,
}: {
  text: string;
  as?: ElementType;
  className?: string;
  start?: number;
}) {
  const [setRef, inView] = useInView<HTMLElement>({ threshold: 0.15 });
  const words = text.split(' ');
  return createElement(
    as,
    { ref: setRef, className: `rw ${inView ? 'is-visible' : ''} ${className}`.trim() },
    words.map((w, i) =>
      createElement(
        'span',
        { className: 'rw__w', key: `${w}-${i}` },
        createElement(
          'span',
          { className: 'rw__i', style: { ['--rw-i' as string]: start + i } },
          w,
        ),
      ),
    ),
  );
}

const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

type CountUpProps = {
  to: number;
  from?: number;
  decimals?: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
};

/** Animates a number from `from` → `to` the first time it scrolls into view. */
export function CountUp({
  to,
  from = 0,
  decimals = 0,
  duration = 1400,
  prefix = '',
  suffix = '',
  className,
}: CountUpProps) {
  const [setRef, inView] = useInView<HTMLSpanElement>({ threshold: 0.6 });
  const [value, setValue] = useState(from);

  useEffect(() => {
    if (!inView) return;
    if (prefersReducedMotion()) {
      setValue(to);
      return;
    }
    let raf = 0;
    let start = 0;
    const tick = (ts: number) => {
      if (!start) start = ts;
      const p = Math.min(1, (ts - start) / duration);
      setValue(from + (to - from) * easeOutCubic(p));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    // Safety net: guarantee the final value even if rAF is throttled/paused.
    const done = window.setTimeout(() => setValue(to), duration + 120);
    return () => {
      cancelAnimationFrame(raf);
      window.clearTimeout(done);
    };
  }, [inView, to, from, duration]);

  const formatted = value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return (
    <span ref={setRef} className={className}>
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
}

/** Reveals its lines one at a time once in view — the terminal playback.
 * Returns how many lines should currently be visible. */
export function useLinePlayback(total: number, stepMs = 260): [(n: HTMLElement | null) => void, number] {
  const [setRef, inView] = useInView<HTMLElement>({ threshold: 0.35 });
  const [shown, setShown] = useState(0);

  useEffect(() => {
    if (!inView) return;
    if (prefersReducedMotion()) {
      setShown(total);
      return;
    }
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setShown(i);
      if (i >= total) window.clearInterval(id);
    }, stepMs);
    return () => window.clearInterval(id);
  }, [inView, total, stepMs]);

  return [setRef, shown];
}

/** Cursor-driven 3D tilt with spring-damped easing.
 *
 * Tracks the pointer anywhere on screen (normalised against the viewport, not
 * just the element) so the card reacts before you reach it, and writes the
 * result to CSS custom properties — `--rx`/`--ry` for rotation, `--gx`/`--gy`
 * for the specular highlight, `--sx`/`--sy` for a shadow that falls opposite
 * the tilt, and `--lift` for proximity-based elevation. CSS does the painting;
 * this only ever touches custom properties, so there is no layout thrash.
 *
 * No-ops for reduced-motion users and on touch-only pointers.
 */
export function useTilt3D<T extends HTMLElement>(maxDeg = 10) {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || prefersReducedMotion()) return;
    if (window.matchMedia?.('(hover: none)').matches) return;

    // target vs current — current chases target for a springy, weighted feel
    let tX = 0, tY = 0, tLift = 0, tGx = 50, tGy = 50;
    let cX = 0, cY = 0, cLift = 0, cGx = 50, cGy = 50;
    let raf = 0;

    const tick = () => {
      const k = 0.09; // damping: lower = heavier, more floaty
      cX += (tX - cX) * k;
      cY += (tY - cY) * k;
      cLift += (tLift - cLift) * k;
      cGx += (tGx - cGx) * k;
      cGy += (tGy - cGy) * k;

      const s = el.style;
      s.setProperty('--rx', `${cX.toFixed(3)}deg`);
      s.setProperty('--ry', `${cY.toFixed(3)}deg`);
      s.setProperty('--lift', cLift.toFixed(4));
      s.setProperty('--gx', `${cGx.toFixed(2)}%`);
      s.setProperty('--gy', `${cGy.toFixed(2)}%`);
      // shadow leans against the tilt, which sells the depth
      s.setProperty('--sx', (-cY * 1.7).toFixed(2));
      s.setProperty('--sy', (cX * 1.7 + 26).toFixed(2));

      const settled =
        Math.abs(tX - cX) < 0.01 &&
        Math.abs(tY - cY) < 0.01 &&
        Math.abs(tLift - cLift) < 0.001 &&
        Math.abs(tGx - cGx) < 0.05 &&
        Math.abs(tGy - cGy) < 0.05;
      raf = settled ? 0 : requestAnimationFrame(tick);
    };
    const wake = () => {
      if (!raf) raf = requestAnimationFrame(tick);
    };

    const onMove = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      if (!r.width) return;
      const cxEl = r.left + r.width / 2;
      const cyEl = r.top + r.height / 2;

      // normalised offset from the card's centre, in viewport halves
      const nx = Math.max(-1, Math.min(1, (e.clientX - cxEl) / (window.innerWidth / 2)));
      const ny = Math.max(-1, Math.min(1, (e.clientY - cyEl) / (window.innerHeight / 2)));

      tY = nx * maxDeg;        // rotateY follows horizontal travel
      tX = -ny * maxDeg;       // rotateX inverts so it tips toward the cursor
      tGx = ((e.clientX - r.left) / r.width) * 100;
      tGy = ((e.clientY - r.top) / r.height) * 100;

      // lift ramps up as the pointer nears the card
      const d = Math.hypot(e.clientX - cxEl, e.clientY - cyEl);
      tLift = Math.max(0, 1 - d / (Math.max(r.width, r.height) * 1.9));
      wake();
    };

    const onLeave = () => {
      tX = 0; tY = 0; tLift = 0; tGx = 50; tGy = 50;
      wake();
    };

    window.addEventListener('pointermove', onMove, { passive: true });
    window.addEventListener('pointerdown', onMove, { passive: true });
    document.addEventListener('pointerleave', onLeave);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerdown', onMove);
      document.removeEventListener('pointerleave', onLeave);
      cancelAnimationFrame(raf);
    };
  }, [maxDeg]);

  return ref;
}

/** Subtle cursor-follow translate for a primary CTA. No-ops on touch and for
 * reduced-motion users. */
export function useMagnetic<T extends HTMLElement>(strength = 0.28) {
  const ref = useRef<T | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || prefersReducedMotion()) return;
    if (window.matchMedia?.('(hover: none)').matches) return;

    const onMove = (e: MouseEvent) => {
      const r = el.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2);
      const dy = e.clientY - (r.top + r.height / 2);
      el.style.transform = `translate(${dx * strength}px, ${dy * strength}px)`;
    };
    const reset = () => {
      el.style.transform = '';
    };
    el.addEventListener('mousemove', onMove);
    el.addEventListener('mouseleave', reset);
    return () => {
      el.removeEventListener('mousemove', onMove);
      el.removeEventListener('mouseleave', reset);
      reset();
    };
  }, [strength]);
  return ref;
}
