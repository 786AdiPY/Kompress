/** Standard error panel. Accepts anything thrown (Error, string, unknown) and
 * surfaces a readable message. */
export default function ErrorState({
  error,
  title = 'Something went wrong',
}: {
  error: unknown;
  title?: string;
}) {
  return (
    <div className="error-state" role="alert">
      <div className="error-state__title">{title}</div>
      <div>{messageOf(error)}</div>
    </div>
  );
}

function messageOf(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  if (error == null) return 'Unknown error.';
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}
