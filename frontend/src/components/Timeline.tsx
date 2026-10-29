'use client';

import { useEffect, useState } from 'react';

import type { RunState } from '@/lib/reducer';
import { type Tone, groupedTimeline } from '@/lib/timeline';

const DOT: Readonly<Record<Tone, string>> = {
  neutral: 'bg-idle',
  running: 'bg-running',
  done: 'bg-done',
  failed: 'bg-failed',
  retry: 'bg-retry',
  edited: 'bg-edited',
  law: 'bg-law',
};

const SHOW_CALLS = 'atlas:timeline-calls';

function clock(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

/** Whether to list every call, remembered per viewer; never fatal where storage is blocked. */
function useShowCalls(): readonly [boolean, (show: boolean) => void] {
  const [show, setShow] = useState(false);
  useEffect(() => {
    try {
      // Read after hydration: the server renders with calls hidden.
      if (window.localStorage.getItem(SHOW_CALLS) === 'true') setShow(true);
    } catch {
      // Storage blocked (a private window): the default it is.
    }
  }, []);
  return [
    show,
    (next) => {
      setShow(next);
      try {
        window.localStorage.setItem(SHOW_CALLS, String(next));
      } catch {
        // Not remembered, still shown.
      }
    },
  ];
}

/**
 * How the run unfolded, step by step. An entry about an agent selects it, so
 * the timeline is also a way in. Each search and model call is folded away
 * unless asked for; the inspector's Attempts tab lists them all.
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
  const [showCalls, setShowCalls] = useShowCalls();
  const groups = groupedTimeline(state, { showCalls });

  if (groups.length === 0) {
    return <p className="px-3 py-3 text-xs text-ink-faint">Waiting for the run to start…</p>;
  }

  return (
    <div className="pb-3">
      <label className="flex cursor-pointer items-center gap-2 px-3 pt-2.5 text-2xs text-ink-muted">
        <input
          type="checkbox"
          checked={showCalls}
          onChange={(event) => setShowCalls(event.target.checked)}
          className="accent-accent"
        />
        Show every call
      </label>
      <ol aria-label="Run timeline">
        {groups.map((group) => (
          <li key={`${group.key}:${group.round}:${group.entries[0]?.seq ?? 0}`}>
            <h3 className="sticky top-0 z-10 bg-surface/95 px-3 pb-1 pt-3 text-2xs font-semibold text-ink-muted backdrop-blur">
              {group.label}
            </h3>
            <ol className="px-2">
              {group.entries.map((entry) => {
                const target = entry.nodeId;
                const active = target !== null && target === selected;
                const body = (
                  <>
                    <span className="w-9 shrink-0 pt-px font-mono text-2xs text-ink-faint">{clock(entry.at)}</span>
                    <span aria-hidden="true" className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${DOT[entry.tone]}`} />
                    <span className="min-w-0 flex-1">
                      <span className="block text-xs text-ink">{entry.text}</span>
                      {entry.detail !== null && (
                        <span className="block whitespace-pre-line break-words text-2xs text-ink-muted">
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
                        className={`flex w-full gap-2 rounded-md px-1.5 py-1 text-left hover:bg-raised ${active ? 'bg-raised' : ''}`}
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
          </li>
        ))}
      </ol>
    </div>
  );
}
