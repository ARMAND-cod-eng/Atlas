'use client';

import Link from 'next/link';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

import { CitationChip } from './CitationChip';
import { LAW_AS_OF } from './Disclaimer';
import { CompareControl, Diff } from './Versions';
import { Badge, Button, Disclosure, buttonClass } from './ui';
import { CopyButton } from './ui/CopyButton';
import { CITE_PREFIX, judgementsOf, markCitations, statusOf, supportFor } from '@/lib/citations';
import { answerMarkdown } from '@/lib/copy';
import { hostOf } from '@/lib/format';
import type { RunResult, SourceRef } from '@/lib/events';
import { byAuthority } from '@/lib/inspector';
import type { RunState, SupportCheck } from '@/lib/reducer';
import { type VersionRow, comparable } from '@/lib/versions';

interface Props {
  readonly state: RunState;
  readonly result: RunResult | null;
}

/** A first line that is only a heading saying "Answer": the card it is shown in says so already. */
const TITLE = /^\s*#{1,6}[ \t]+answer[ \t]*#*[ \t]*(?:\r?\n)+/i;

/**
 * The answer, as the thing the page is for: a card with the answer at reading
 * size, the verdict of the checks under it, and the two things to do with it,
 * copy it or compare it with another version, in its header. Comparing swaps
 * the answer for the changes in place, and back.
 */
export function AnswerCard({
  state,
  result,
  rows,
  onResearchAnyway,
}: Props & {
  readonly rows: readonly VersionRow[];
  /**
   * Research a declined question after all; resolves to why it could not start,
   * or null. Absent on a replay, which has nothing to research it with.
   */
  readonly onResearchAnyway?: () => Promise<string | null>;
}): React.JSX.Element {
  const [against, setAgainst] = useState<string | null>(null);
  const options = comparable(rows);
  const chosen = options.find((row) => row.runId === against) ?? null;

  if (state.declined !== null) return <Declined reason={state.declined} onResearchAnyway={onResearchAnyway} />;
  if (state.answer === null) return <Pending state={state} />;
  const answer = state.answer;
  // Every source the run had: the result's once it has one, the branches' as they land.
  const sources = [...(result?.sources ?? []), ...state.branches.flatMap((branch) => branch.sources)];

  return (
    <section aria-labelledby="answer-title" className="rounded-xl border border-line bg-surface p-4 shadow-sm sm:p-5">
      <header className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-2">
        <h2 id="answer-title" className="text-sm font-semibold text-ink">
          {chosen === null ? 'Answer' : `Changes since v${chosen.number}`}
        </h2>
        <div className="ml-auto flex flex-wrap items-center gap-1.5">
          {chosen !== null && (
            <Button variant="ghost" size="sm" onClick={() => setAgainst(null)}>
              ← Back to answer
            </Button>
          )}
          <CompareControl options={options} chosen={against} onChoose={setAgainst} />
          <CopyButton
            label="Copy"
            title="Copy as Markdown, with its citations"
            text={() => answerMarkdown({ question: state.query, answer, sources, lawAsOf: LAW_AS_OF })}
          />
        </div>
      </header>
      {chosen === null ? (
        <Answer state={state} result={result} />
      ) : (
        <Diff key={chosen.runId} chosen={chosen} answer={answer} />
      )}
    </section>
  );
}

/**
 * A question judged to be about neither act. Said in place of an answer, since
 * nothing was searched or cited: an answer here would be the nearest-sounding
 * law, marked verified because it exists.
 */
function Declined({
  reason,
  onResearchAnyway,
}: {
  readonly reason: string;
  readonly onResearchAnyway?: () => Promise<string | null>;
}): React.JSX.Element {
  const [starting, setStarting] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  async function research(): Promise<void> {
    if (onResearchAnyway === undefined || starting) return;
    setStarting(true);
    setFailure(null);
    const why = await onResearchAnyway();
    if (why !== null) {
      setFailure(why);
      setStarting(false);
    }
  }

  return (
    <section aria-labelledby="answer-title" className="rounded-xl border border-retry/40 bg-retry/5 p-4 sm:p-5">
      <h2 id="answer-title" className="text-sm font-semibold text-ink">
        Not about the AI Act or the GDPR
      </h2>
      <p className="mt-2 max-w-[68ch] text-base leading-relaxed text-ink">{reason}</p>
      <p className="mt-2 max-w-[68ch] text-xs leading-relaxed text-ink-muted">
        Atlas researches the EU AI Act and the GDPR, so this question was not researched: nothing
        was searched, and nothing in the law was cited for it.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Link href="/" className={buttonClass('primary', 'sm')}>
          Ask another question
        </Link>
        {onResearchAnyway !== undefined && (
          <Button variant="secondary" size="sm" disabled={starting} onClick={() => void research()}>
            {starting ? 'Starting…' : 'Research anyway'}
          </Button>
        )}
        {failure !== null && (
          <p role="alert" className="text-2xs text-failed">
            {failure}
          </p>
        )}
      </div>
    </section>
  );
}

/** Where the answer will be, said honestly while there is none. */
function Pending({ state }: { readonly state: RunState }): React.JSX.Element {
  if (state.phase === 'failed') {
    return (
      <section className="rounded-xl border border-failed/40 bg-failed/5 p-4 sm:p-5">
        <h2 className="text-sm font-semibold text-ink">No answer</h2>
        <p className="mt-1 text-xs text-ink-muted">
          The run failed before an answer was written{state.error !== null ? `: ${state.error}` : '.'}
        </p>
      </section>
    );
  }
  return (
    <section aria-busy="true" className="rounded-xl border border-line bg-surface p-4 sm:p-5">
      <h2 className="text-sm font-semibold text-ink">Answer</h2>
      <p className="mt-1 text-xs text-ink-muted">
        Being written. It appears here once every branch is researched and the critic has read it
        against the sources; what each branch found is below as it lands.
      </p>
      <div aria-hidden="true" className="mt-4 space-y-2">
        {['w-full', 'w-11/12', 'w-4/5'].map((width) => (
          <div key={width} className={`h-3 rounded bg-raised motion-safe:animate-pulse ${width}`} />
        ))}
      </div>
    </section>
  );
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
export function Answer({ state }: Props): React.JSX.Element | null {
  if (state.answer === null) return null;

  const failedBranches = state.branches.filter((branch) => branch.status === 'failed');
  const { markdown, labels } = markCitations(state.answer.replace(TITLE, ''));
  const check = state.citationChecks.at(-1) ?? null;
  // Only the judgement of the draft on screen: an earlier draft's says nothing about this one.
  const support = supportFor(state.citationChecks, state.supportChecks);

  return (
    <div className="space-y-4">
      {failedBranches.length > 0 && (
        // Said before the answer, not after it. An answer built on four fifths of
        // the research reads as complete unless something says otherwise.
        <p className="rounded-md border border-retry/40 bg-retry/10 px-3 py-2 text-xs text-retry">
          {failedBranches.length} of {state.branches.length} research branches failed. This answer is
          based on the {state.branches.length - failedBranches.length} that succeeded.
        </p>
      )}

      <article className="prose-atlas max-w-[68ch] text-base leading-relaxed text-ink">
        <ReactMarkdown
          components={{
            a: ({ href, children }) => {
              if (href?.startsWith(CITE_PREFIX) === true) {
                const label = labels[Number(href.slice(CITE_PREFIX.length))];
                if (label !== undefined) {
                  return (
                    <CitationChip
                      label={label}
                      status={statusOf(label, check, support)}
                      judgements={judgementsOf(label, support)}
                    />
                  );
                }
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

      {check !== null && <Verdict check={check} support={support} />}
    </div>
  );
}

/** What the checks made of the answer's citations: a badge to see at a glance, a sentence to read. */
function Verdict({
  check,
  support,
}: {
  readonly check: NonNullable<RunState['citationChecks'][number]>;
  readonly support: SupportCheck | null;
}): React.JSX.Element {
  const problems = check.unresolved.length + check.unretrieved.length + check.misquoted.length;
  const unsupported = support?.unsupported.length ?? 0;
  const judged = support === null ? 0 : support.supported.length + unsupported + support.unverified.length;
  const found = problems + unsupported;
  return (
    <div className="flex flex-wrap items-start gap-x-2.5 gap-y-1.5 border-t border-line pt-3">
      <Badge tone={found === 0 ? 'done' : 'failed'}>
        {found === 0 ? (
          <>
            <span aria-hidden="true">✓</span> Citations verified
          </>
        ) : (
          `${found} citation ${found === 1 ? 'problem' : 'problems'}`
        )}
      </Badge>
      <p className="min-w-0 flex-1 basis-60 text-2xs leading-relaxed text-ink-muted">
        {problems === 0
          ? check.cited.length === 1
            ? 'The citation was checked against the law: it exists and was retrieved.'
            : `All ${check.cited.length} citations checked against the law: each exists and was retrieved.`
          : `${problems} of ${check.cited.length} citations failed the check against the law. Hover a chip for why.`}
        {support !== null &&
          judged > 0 &&
          (unsupported === 0
            ? judged === 1
              ? ' The support check read it against its provision: it says what it is cited for.'
              : ` The support check read ${judged} against their provisions: each says what it is cited for.`
            : ` The support check read ${judged} against their provisions: ${unsupported} ${unsupported === 1 ? 'does' : 'do'} not say what ${unsupported === 1 ? 'it is' : 'they are'} cited for.`)}
      </p>
    </div>
  );
}

/** Every source the answer could draw on, the law first, folded away until wanted. */
export function Sources({ result }: { readonly result: RunResult | null }): React.JSX.Element | null {
  const sources = result?.sources ?? [];
  if (sources.length === 0) return null;
  const grouped = byAuthority(sources);
  return (
    <Disclosure title="Sources" count={sources.length}>
      <div className="space-y-4">
        <SourceGroup title="Primary law" tone="text-law" sources={grouped.law} />
        <SourceGroup title="Official guidance" tone="text-guidance" sources={grouped.guidance} />
        <SourceGroup title="Other sources" tone="text-secondary" sources={grouped.other} />
      </div>
    </Disclosure>
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
      <h3 className="text-2xs font-semibold text-ink-muted">
        {title} · {sources.length}
      </h3>
      <ol className="mt-1.5 space-y-1">
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
              {/* A recital's title is its label: said once. */}
              {source.title !== source.label && source.title}
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
