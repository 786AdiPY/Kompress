import type { RunStatus } from '../api/types';

const LABELS: Record<RunStatus, string> = {
  pending_gate: 'Pending Gate',
  pending_approval: 'Pending Approval',
  approved: 'Approved',
  rejected: 'Rejected',
  failed: 'Failed',
};

/** Colored status pill. Accepts a known RunStatus or any string (falls back to
 * a neutral chip and a title-cased label). */
export default function StatusChip({ status }: { status: RunStatus | string }) {
  const label = LABELS[status as RunStatus] ?? prettify(status);
  return <span className={`chip chip--${status}`}>{label}</span>;
}

function prettify(s: string): string {
  return s
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(' ');
}
