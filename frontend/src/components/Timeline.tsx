'use client';

import type { RunState } from '@/lib/reducer';
import { type Tone, timelineEntries } from '@/lib/timeline';

const DOT: Readonly<Record<Tone, string>> = {
  neutral: 'bg-idle',
  running: 'bg-running',
  done: 'bg-done',
  failed: 'bg-failed',
  retry: 'bg-retry',
  edited: 'bg-edited',
  law: 'bg-law',
};

function clock(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

/**
 * How the run unfolded, one line per thing that happened. An entry about an
 * agent selects it in the graph, so the timeline is also a way in.
 */
export function Timeline({
  state,
  selected = null,
  onSelect,
}: {
  readonly state: RunState;
  readonly selected?: string | null;
  readonly onSelect?: (nodeId: string) => void;
}): React.JSX.Element {
  const entries = timelineEntries(state);

  if (entries.length === 0) {
    return <p className="px-3 py-3 text-xs text-ink-faint">Waiting for the run to start…</p>;
  }

  return (
    <ol className="px-2 py-2" aria-label="Run timeline">
      {entries.map((entry) => {
        const target = entry.nodeId;
        const active = target !== null && target === selected;
        const body = (
          <>
            <span className="w-9 shrink-0 pt-px font-mono text-2xs text-ink-faint">{clock(entry.at)}</span>
            <span aria-hidden="true" className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${DOT[entry.tone]}`} />
            <span className="min-w-0 flex-1">
              <span className="block text-xs text-ink">{entry.text}</span>
              {entry.detail !== null && (
                <span className="block whitespace-pre-line break-words text-2xs text-ink-faint">
                  {entry.detail}
                </span>
              )}
            </span>
          </>
        );
        return (
          <li key={entry.seq}>
            {target !== null && onSelect !== undefined ? (
              <button
                type="button"
                onClick={() => onSelect(target)}
                aria-pressed={active}
                className={`flex w-full gap-2 rounded px-1.5 py-1 text-left hover:bg-raised ${active ? 'bg-raised' : ''}`}
              >
                {body}
              </button>
            ) : (
              <div className="flex gap-2 px-1.5 py-1">{body}</div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
