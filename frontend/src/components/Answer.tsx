'use client';

import ReactMarkdown from 'react-markdown';

import { CitationChip } from './CitationChip';
import { CITE_PREFIX, markCitations, statusOf } from '@/lib/citations';
import { hostOf } from '@/lib/format';
import type { RunResult, SourceRef } from '@/lib/events';
import { byAuthority } from '@/lib/inspector';
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
 *
 * Each citation of the law is a chip carrying the verdict of the run's last
 * citation check, the one made on the answer shown.
 */
export function Answer({ state, result }: Props): React.JSX.Element | null {
  if (state.answer === null) return null;

  const failedBranches = state.branches.filter((branch) => branch.status === 'failed');
  const { markdown, labels } = markCitations(state.answer);
  const check = state.citationChecks.at(-1) ?? null;
  const grouped = byAuthority(result?.sources ?? []);

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
            a: ({ href, children }) => {
              if (href?.startsWith(CITE_PREFIX) === true) {
                const label = labels[Number(href.slice(CITE_PREFIX.length))];
                if (label !== undefined) return <CitationChip label={label} status={statusOf(label, check)} />;
              }
              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noreferrer"
                  className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
                >
                  {children}
                </a>
              );
            },
          }}
        >
          {markdown}
        </ReactMarkdown>
      </article>

      {check !== null && <CheckSummary check={check} />}

      <SourceGroup title="Primary law" tone="text-law" sources={grouped.law} />
      <SourceGroup title="Official guidance" tone="text-guidance" sources={grouped.guidance} />
      <SourceGroup title="Other sources" tone="text-secondary" sources={grouped.other} />
    </section>
  );
}

function CheckSummary({ check }: { readonly check: NonNullable<RunState['citationChecks'][number]> }): React.JSX.Element {
  const problems = check.unresolved.length + check.unretrieved.length + check.misquoted.length;
  return (
    <p className={`text-2xs ${problems === 0 ? 'text-ink-faint' : 'text-retry'}`}>
      {problems === 0
        ? `All ${check.cited.length} citations checked against the law: each exists and was retrieved.`
        : `${problems} of ${check.cited.length} citations failed the check against the law. Hover a chip for why.`}
    </p>
  );
}

function SourceGroup({
  title,
  tone,
  sources,
}: {
  readonly title: string;
  readonly tone: string;
  readonly sources: readonly SourceRef[];
}): React.JSX.Element | null {
  if (sources.length === 0) return null;
  return (
    <div>
      <h2 className="text-2xs uppercase tracking-wide text-ink-faint">
        {title} · {sources.length}
      </h2>
      <ol className="mt-2 space-y-1">
        {sources.map((source, index) => (
          // Keyed by position too: two paragraphs of one article share its URL.
          <li key={`${source.label ?? source.url}:${index}`} className="flex gap-2 text-xs">
            <a
              href={source.url}
              target="_blank"
              rel="noreferrer"
              className="min-w-0 truncate text-ink hover:text-accent"
              title={source.url}
            >
              {source.label !== null && source.label !== undefined && (
                <span className={`mr-1.5 font-medium ${tone}`}>{source.label}</span>
              )}
              {source.title}
              {(source.label === null || source.label === undefined) && (
                <span className="ml-1.5 text-ink-faint">{hostOf(source.url)}</span>
              )}
            </a>
          </li>
        ))}
      </ol>
    </div>
  );
}
