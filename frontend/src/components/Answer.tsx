'use client';

import ReactMarkdown from 'react-markdown';

import { hostOf } from '@/lib/format';
import type { RunResult } from '@/lib/events';
import type { RunState } from '@/lib/reducer';

interface Props {
  readonly state: RunState;
  readonly result: RunResult | null;
}

/**
 * The answer and its citations.
 *
 * The answer arrives whole at `run_completed` — Phase 3 deliberately did not add
 * token streaming, so there is nothing to reveal progressively and no pretence
 * that there is. Sources come from `GET /runs/{id}`, which is why the terminal
 * event carries a count rather than the list: replaying every source on each
 * reconnect would make reconnection the most expensive thing a client does.
 */
export function Answer({ state, result }: Props): React.JSX.Element | null {
  if (state.answer === null) return null;

  const sources = result?.sources ?? [];
  const failedBranches = state.branches.filter((branch) => branch.status === 'failed');

  return (
    <section className="space-y-4">
      {failedBranches.length > 0 && (
        // Said before the answer, not after it. An answer built on four fifths of
        // the research reads as complete unless something says otherwise.
        <p className="rounded border border-retry/40 bg-retry/10 px-3 py-2 text-xs text-retry">
          {failedBranches.length} of {state.branches.length} research branches failed. This answer is
          based on the {state.branches.length - failedBranches.length} that succeeded.
        </p>
      )}

      <article className="prose-atlas max-w-none text-sm leading-relaxed text-ink">
        <ReactMarkdown
          components={{
            a: ({ href, children }) => (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-running underline decoration-sky-800 underline-offset-2 hover:decoration-sky-500"
              >
                {children}
              </a>
            ),
          }}
        >
          {state.answer}
        </ReactMarkdown>
      </article>

      {sources.length > 0 && (
        <div>
          <h2 className="text-2xs uppercase tracking-wide text-ink-faint">
            sources · {sources.length}
          </h2>
          <ol className="mt-2 space-y-1">
            {sources.map((source, index) => (
              <li key={source.url} className="flex gap-2 text-xs">
                <span className="shrink-0 tabular-nums text-ink-faint">{index + 1}.</span>
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer"
                  className="truncate text-ink hover:text-accent"
                  title={source.url}
                >
                  {source.title}
                  <span className="ml-1.5 text-ink-faint">{hostOf(source.url)}</span>
                </a>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
