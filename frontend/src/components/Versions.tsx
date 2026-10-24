'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import { fetchAnswer } from '@/lib/api';
import { type VersionRow, answerDiff, comparable } from '@/lib/versions';

const DOT: Record<VersionRow['status'], string> = {
  running: 'bg-running animate-pulse',
  completed: 'bg-done',
  failed: 'bg-failed',
};

/** Every version of the answer on this thread; the one on screen marked. */
export function Versions({ rows }: { readonly rows: readonly VersionRow[] }): React.JSX.Element | null {
  if (rows.length === 0) return null;
  return (
    <section aria-label="Versions" className="border-b border-line p-3">
      <h3 className="mb-1.5 text-2xs uppercase tracking-wide text-ink-faint">Versions · {rows.length}</h3>
      <ol className="space-y-1">
        {rows.map((row) => {
          const body = (
            <>
              <span className="flex items-center gap-2">
                <span className="font-mono text-2xs text-ink-faint">v{row.number}</span>
                <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${DOT[row.status]}`} aria-label={row.status} />
                <span className="text-xs font-medium text-ink">{row.title}</span>
                {row.basedOn !== null && (
                  <span className="text-2xs text-ink-faint">
                    {row.kind === 'revision' ? 'revises' : 'follows'} v{row.basedOn}
                  </span>
                )}
              </span>
              <span className="mt-0.5 line-clamp-2 block pl-6 text-2xs text-ink-muted">{row.detail}</span>
            </>
          );
          const shape = 'block rounded px-2 py-1.5';
          return (
            <li key={row.runId}>
              {row.current ? (
                <div aria-current="page" className={`${shape} bg-accent/10 ring-1 ring-accent/40`}>
                  {body}
                </div>
              ) : (
                <Link href={`/runs/${row.runId}`} title={row.before ?? undefined} className={`${shape} hover:bg-raised`}>
                  {body}
                </Link>
              )}
            </li>
          );
        })}
      </ol>
      {rows.length === 1 && (
        <p className="mt-2 text-2xs text-ink-faint">
          Open a researcher and rewrite its question to make a new version.
        </p>
      )}
    </section>
  );
}

type Fetched = { readonly runId: string; readonly answer: string | null };

/** This answer against another version's, word by word. */
export function Compare({
  rows,
  answer,
}: {
  readonly rows: readonly VersionRow[];
  readonly answer: string | null;
}): React.JSX.Element | null {
  const [against, setAgainst] = useState<string | null>(null);
  const [fetched, setFetched] = useState<Fetched | null>(null);

  useEffect(() => {
    if (against === null) return;
    let live = true;
    void fetchAnswer(against).then((other) => {
      if (live) setFetched({ runId: against, answer: other });
    });
    return () => {
      live = false;
    };
  }, [against]);

  const options = comparable(rows);
  if (answer === null || options.length === 0) return null;
  const chosen = options.find((row) => row.runId === against) ?? null;

  return (
    <section className="space-y-2">
      <label className="flex items-center gap-2 text-2xs text-ink-faint">
        Compare with
        <select
          value={against ?? ''}
          onChange={(event) => setAgainst(event.target.value === '' ? null : event.target.value)}
          className="rounded border border-line bg-canvas px-1.5 py-0.5 text-2xs text-ink"
        >
          <option value="">—</option>
          {options.map((row) => (
            <option key={row.runId} value={row.runId}>
              v{row.number} · {row.title}
            </option>
          ))}
        </select>
      </label>
      {chosen !== null && <Diff chosen={chosen} fetched={fetched} answer={answer} />}
    </section>
  );
}

function Diff({
  chosen,
  fetched,
  answer,
}: {
  readonly chosen: VersionRow;
  readonly fetched: Fetched | null;
  readonly answer: string;
}): React.JSX.Element {
  if (fetched?.runId !== chosen.runId) return <p className="text-2xs text-ink-faint">Loading v{chosen.number}…</p>;
  if (fetched.answer === null) {
    return <p className="text-2xs text-failed">The API could not return v{chosen.number}’s answer.</p>;
  }
  const { parts, added, removed } = answerDiff(fetched.answer, answer);
  return (
    <div className="rounded border border-line bg-canvas p-3">
      <p className="mb-2 text-2xs text-ink-faint">
        Since v{chosen.number}: <span className="text-done">+{added} words</span>,{' '}
        <span className="text-failed">−{removed} words</span>
      </p>
      <p className="whitespace-pre-wrap text-xs leading-relaxed text-ink">
        {parts.map((part, index) =>
          part.kind === 'same' ? (
            <span key={index}>{part.text}</span>
          ) : part.kind === 'added' ? (
            <ins key={index} className="bg-done/15 text-done no-underline">
              {part.text}
            </ins>
          ) : (
            <del key={index} className="bg-failed/10 text-failed">
              {part.text}
            </del>
          ),
        )}
      </p>
    </div>
  );
}
