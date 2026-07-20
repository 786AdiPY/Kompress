export interface DeltaStatProps {
  label: string;
  value: number | null | undefined;
  unit?: string;
  /** Which sign of `value` should be colored as good.
   * 'negative' => smaller/lower is better (size, latency).
   * 'positive' => larger/higher is better (accuracy, auc, f1, speedup). */
  goodWhen: 'negative' | 'positive';
}

/** A single labeled delta metric, colored good/bad by the sign convention.
 * A null/undefined value renders as an em dash with neutral color. */
export default function DeltaStat({ label, value, unit, goodWhen }: DeltaStatProps) {
  // Tone follows the ROUNDED display value so a delta that renders as "0"
  // never shows in good/bad color (display and tone always agree).
  const rounded = roundForDisplay(value);
  const tone = toneFor(rounded, goodWhen);
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className={`stat__value stat__value--${tone}`}>{format(rounded, unit)}</span>
    </div>
  );
}

function roundForDisplay(value: number | null | undefined): number | null {
  if (value == null || Number.isNaN(value)) return null;
  return Math.abs(value) >= 100 ? Math.round(value) : Math.round(value * 100) / 100;
}

function toneFor(
  value: number | null,
  goodWhen: 'negative' | 'positive',
): 'good' | 'bad' | 'neutral' {
  if (value == null || value === 0) return 'neutral';
  const isPositive = value > 0;
  const good = goodWhen === 'positive' ? isPositive : !isPositive;
  return good ? 'good' : 'bad';
}

function format(rounded: number | null, unit?: string): string {
  if (rounded == null) return '—';
  const sign = rounded > 0 ? '+' : '';
  return `${sign}${rounded}${unit ?? ''}`;
}
