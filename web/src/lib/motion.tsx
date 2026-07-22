// Small, dependency-free scroll-motion primitives for the landing page.
// All respect prefers-reduced-motion: when the user opts out, reveals show
// immediately and counters jump straight to their target.
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
