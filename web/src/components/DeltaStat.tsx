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
  const tone = toneFor(value, goodWhen);
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className={`stat__value stat__value--${tone}`}>{format(value, unit)}</span>
    </div>
  );
}

function toneFor(
  value: number | null | undefined,
  goodWhen: 'negative' | 'positive',
): 'good' | 'bad' | 'neutral' {
  if (value == null || Number.isNaN(value) || value === 0) return 'neutral';
  const isPositive = value > 0;
  const good = goodWhen === 'positive' ? isPositive : !isPositive;
  return good ? 'good' : 'bad';
}

function format(value: number | null | undefined, unit?: string): string {
  if (value == null || Number.isNaN(value)) return '—';
  const rounded = Math.abs(value) >= 100 ? Math.round(value) : Math.round(value * 100) / 100;
  const sign = rounded > 0 ? '+' : '';
  return `${sign}${rounded}${unit ?? ''}`;
}
