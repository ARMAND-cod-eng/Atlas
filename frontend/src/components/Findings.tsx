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
      <h2 className="text-[11px] uppercase tracking-wide text-neutral-500">research</h2>
      <ul className="mt-2 space-y-3">
        {state.branches.map((branch) => (
          <li key={branch.id}>
            <div className="flex items-baseline gap-2">
              <span className="shrink-0 font-mono text-[10px] text-neutral-600">
                {branch.subtaskId}
              </span>
              <p className="text-[12px] leading-snug text-neutral-300">{branch.question}</p>
            </div>

            {branch.status === 'failed' ? (
              <p className="mt-1 pl-7 text-[11px] text-red-400">
                {branch.error ?? 'this branch failed'} — the answer below does not include it
              </p>
            ) : branch.sources.length === 0 ? (
              <p className="mt-1 pl-7 text-[11px] text-neutral-600">
                {branch.status === 'running' ? 'searching…' : 'no sources'}
              </p>
            ) : (
              <ul className="mt-1 space-y-0.5 pl-7">
                {branch.sources.map((source) => (
                  <li key={source.url} className="truncate text-[11px]">
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-neutral-400 hover:text-sky-300"
                      title={source.title}
                    >
                      {source.title}
                    </a>
                    <span className="ml-1.5 text-neutral-600">{hostOf(source.url)}</span>
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
