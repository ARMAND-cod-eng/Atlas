/**
 * Small pure formatters. Kept out of components so they can be read — and, where
 * it matters, tested — without rendering anything.
 */

/** Seconds elapsed between an ISO timestamp and a millisecond clock. */
export function elapsedSeconds(startedAt: string | null, now: number): number | null {
  if (startedAt === null) return null;
  const started = Date.parse(startedAt);
  if (Number.isNaN(started)) return null;
  return Math.max(0, (now - started) / 1000);
}

/** Seconds between two ISO timestamps. */
export function durationSeconds(startedAt: string | null, finishedAt: string | null): number | null {
  if (startedAt === null || finishedAt === null) return null;
  const started = Date.parse(startedAt);
  const finished = Date.parse(finishedAt);
  if (Number.isNaN(started) || Number.isNaN(finished)) return null;
  return Math.max(0, (finished - started) / 1000);
}

export function formatSeconds(value: number | null): string {
  if (value === null) return '—';
  return `${value.toFixed(1)}s`;
}

export function formatTokens(value: number | null): string {
  if (value === null) return '—';
  return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(value);
}

/** `researcher_failed:s1` reads better as `researcher failed · s1`. */
export function formatDegraded(flag: string): string {
  const [name, detail] = flag.split(':');
  const label = (name ?? flag).replace(/_/g, ' ');
  return detail === undefined ? label : `${label} · ${detail}`;
}

export function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}
