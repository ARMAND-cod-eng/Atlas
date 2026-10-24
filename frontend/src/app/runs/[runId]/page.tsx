'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { use } from 'react';

import { QuestionForm } from '@/components/QuestionForm';
import { RunWorkspace } from '@/components/RunWorkspace';
import { ThemeToggle } from '@/components/ThemeToggle';
import { isFinished } from '@/lib/reducer';
import { useNow, useRunStream, useTopology } from '@/lib/useRunStream';

/**
 * One run, in the three-pane workspace.
 *
 * Deep-linked on purpose. The run belongs to the worker, not to this tab, so its
 * URL survives a refresh: reloading replays the event log from `seq 0` and
 * rejoins live if the run is still going — from Postgres's archive once Redis
 * has let it go.
 *
 * Once the run ends, a follow-up continues the same thread — planned against
 * this answer — and links back here with `?from=`, so a conversation can be
 * walked back one question at a time.
 */
export default function RunPage({
  params,
  searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ from?: string | string[] }>;
}): React.JSX.Element {
  const { runId } = use(params);
  const { from } = use(searchParams);
  const previous = typeof from === 'string' ? from : null;
  const router = useRouter();
  const topology = useTopology();
  const { state, connection, result, missing } = useRunStream(runId);
  const finished = isFinished(state);
  const now = useNow(!finished && !missing);

  if (missing) {
    return (
      <main className="mx-auto flex min-h-screen w-full max-w-2xl flex-col justify-center px-6">
        <h1 className="text-sm font-medium tracking-wide text-ink">No run here</h1>
        <p className="mt-2 text-sm leading-relaxed text-ink-muted">
          The API has no run <span className="font-mono text-ink">{runId}</span>. It may have been
          deleted with its conversation, or the id is wrong.
        </p>
        <Link
          href="/"
          className="mt-6 self-start rounded border border-line-strong px-3 py-1.5 text-xs text-ink hover:border-ink-faint"
        >
          Ask a question
        </Link>
      </main>
    );
  }

  const followUp =
    finished && state.threadId !== null ? (
      <section>
        <h2 className="mb-2 text-2xs uppercase tracking-wide text-ink-faint">follow up</h2>
        <QuestionForm
          threadId={state.threadId}
          parentRunId={runId}
          placeholder="Ask a follow-up. It is planned against this answer."
          hint="continues this conversation"
          submitLabel="Ask"
          onStarted={(next) => router.push(`/runs/${next}?from=${runId}`)}
        />
      </section>
    ) : null;

  return (
    <main className="flex h-screen flex-col bg-canvas text-ink">
      <header className="flex shrink-0 items-center gap-3 border-b border-line bg-surface px-4 py-2">
        <Link href="/" className="text-sm font-medium text-ink hover:text-accent">
          Atlas
        </Link>
        {previous !== null && (
          <Link href={`/runs/${previous}`} className="shrink-0 text-xs text-ink-faint hover:text-ink">
            ← previous question
          </Link>
        )}
        <p className="min-w-0 flex-1 truncate text-xs text-ink-muted">{state.query ?? 'loading run…'}</p>
        <span className="hidden shrink-0 font-mono text-2xs text-ink-faint sm:inline">{runId}</span>
        <ThemeToggle />
      </header>
      <RunWorkspace
        runId={runId}
        state={state}
        result={result}
        topology={topology}
        now={now}
        connection={connection}
        followUp={followUp}
        onRevised={(next) => router.push(`/runs/${next}`)}
      />
    </main>
  );
}
