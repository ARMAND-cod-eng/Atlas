'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import { Button } from './ui';
import { fetchAnswer } from '@/lib/api';
import { type VersionRow, answerDiff } from '@/lib/versions';

const DOT: Record<VersionRow['status'], string> = {
  running: 'bg-running animate-pulse',
  completed: 'bg-done',
  failed: 'bg-failed',
};

/** Every version of the answer on this thread; the one on screen marked. */
export function Versions({ rows }: { readonly rows: readonly VersionRow[] }): React.JSX.Element | null {
  if (rows.length === 0) return null;
  if (rows.length === 1) {
    // One version is not a list: say how to make a second instead.
    return (
      <p className="border-b border-line px-3 py-2 text-2xs text-ink-muted">
        Rewrite a researcher&apos;s question to make a new version and compare the two.
      </p>
    );
  }
  return (
    <section aria-label="Versions" className="border-b border-line p-3">
      <h3 className="mb-1.5 text-2xs font-semibold text-ink-muted">
        Versions <span className="font-normal text-ink-faint">{rows.length}</span>
      </h3>
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
    </section>
  );
}

type Fetched = { readonly runId: string; readonly answer: string | null };

/**
 * The versions this answer can be compared with, the one it revises first. A
 * button each, pressed while its comparison is shown.
 */
export function CompareControl({
  options,
  chosen,
  onChoose,
}: {
  readonly options: readonly VersionRow[];
  readonly chosen: string | null;
  readonly onChoose: (runId: string) => void;
}): React.JSX.Element | null {
  if (options.length === 0) return null;
  return (
    <div role="group" aria-label="Compare with" className="flex items-center gap-1">
      <span aria-hidden="true" className="text-2xs text-ink-muted">
        Compare with
      </span>
      {options.map((row) => (
        <Button
          key={row.runId}
          variant="secondary"
          size="sm"
          aria-label={`Compare with v${row.number}`}
          aria-pressed={chosen === row.runId}
          title={row.title}
          onClick={() => onChoose(row.runId)}
          className={chosen === row.runId ? '!border-accent !text-accent' : ''}
        >
          v{row.number}
        </Button>
      ))}
    </div>
  );
}

/**
 * This answer against another version's, word by word. Fetched for the version
 * named, and shown only under its name: mounted per version, so the changes of
 * one are never on screen while another loads.
 */
export function Diff({
  chosen,
  answer,
}: {
  readonly chosen: VersionRow;
  readonly answer: string;
}): React.JSX.Element {
  const [fetched, setFetched] = useState<Fetched | null>(null);

  useEffect(() => {
    let live = true;
    void fetchAnswer(chosen.runId).then((other) => {
      if (live) setFetched({ runId: chosen.runId, answer: other });
    });
    return () => {
      live = false;
    };
  }, [chosen.runId]);

  if (fetched?.runId !== chosen.runId) return <p className="text-xs text-ink-faint">Loading v{chosen.number}…</p>;
  if (fetched.answer === null) {
    return <p className="text-xs text-failed">The API could not return v{chosen.number}’s answer.</p>;
  }
  const { parts, added, removed } = answerDiff(fetched.answer, answer);
  return (
    <div>
      <p className="mb-3 text-2xs text-ink-muted">
        Since v{chosen.number}: <span className="text-done">+{added} words</span>,{' '}
        <span className="text-failed">−{removed} words</span>
      </p>
      <p className="max-w-[68ch] whitespace-pre-wrap text-base leading-relaxed text-ink">
        {parts.map((part, index) =>
          part.kind === 'same' ? (
            <span key={index}>{part.text}</span>
          ) : part.kind === 'added' ? (
            <ins key={index} className="rounded-sm bg-done/15 text-done no-underline">
              {part.text}
            </ins>
          ) : (
            <del key={index} className="rounded-sm bg-failed/10 text-failed">
              {part.text}
            </del>
          ),
        )}
      </p>
    </div>
  );
}
