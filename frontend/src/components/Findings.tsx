'use client';

import { hostOf } from '@/lib/format';
import type { RunState } from '@/lib/reducer';

interface Props {
  readonly state: RunState;
}

/**
 * The sources each branch retrieved, as they land.
 *
 * This is the other half of the answer to the silent stretch. By the time the
 * synthesizer starts, every researcher has finished and its sources exist — the
 * event stream simply used to throw them away and report a count at the end.
 * Showing them is not progress indication; it is work already done that was
 * previously hidden, and it gives the user something real to read for the
 * 20–45 seconds during which nothing else can honestly be said.
 */
export function Findings({ state }: Props): React.JSX.Element | null {
  if (state.branches.length === 0) return null;

  return (
    <section>
      <h2 className="text-2xs uppercase tracking-wide text-ink-faint">research</h2>
      <ul className="mt-2 space-y-3">
        {state.branches.map((branch) => (
          <li key={branch.id}>
            <div className="flex items-baseline gap-2">
              <span className="shrink-0 font-mono text-2xs text-ink-faint">
                {branch.subtaskId}
              </span>
              <p className="text-xs leading-snug text-ink">{branch.question}</p>
            </div>

            {branch.status === 'failed' ? (
              <p className="mt-1 pl-7 text-2xs text-failed">
                {branch.error ?? 'this branch failed'} — the answer below does not include it
              </p>
            ) : branch.sources.length === 0 ? (
              <p className="mt-1 pl-7 text-2xs text-ink-faint">
                {branch.status === 'running' ? 'searching…' : 'no sources'}
              </p>
            ) : (
              <ul className="mt-1 space-y-0.5 pl-7">
                {branch.sources.map((source, index) => (
                  // Keyed by position too: two paragraphs of one article share its URL.
                  <li key={`${source.label ?? source.url}:${index}`} className="truncate text-2xs">
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-ink-muted hover:text-accent"
                      title={source.title}
                    >
                      {source.label !== null && source.label !== undefined && (
                        <span className="mr-1.5 font-medium text-law">{source.label}</span>
                      )}
                      {source.title}
                    </a>
                    {(source.label === null || source.label === undefined) && (
                      <span className="ml-1.5 text-ink-faint">{hostOf(source.url)}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
