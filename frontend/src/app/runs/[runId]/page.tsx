'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { use } from 'react';

import { Answer } from '@/components/Answer';
import { Findings } from '@/components/Findings';
import { GraphCanvas } from '@/components/GraphCanvas';
import { Metrics } from '@/components/Metrics';
import { QuestionForm } from '@/components/QuestionForm';
import { RunStatus } from '@/components/RunStatus';
import { isFinished } from '@/lib/reducer';
import { useNow, useRunStream, useTopology } from '@/lib/useRunStream';

/**
 * One run.
 *
 * Deep-linked on purpose. The run belongs to the worker, not to this tab, so its
 * URL survives a refresh: reloading replays the event log from `seq 0` and
 * rejoins live if the run is still going. That is Phase 3's event log doing
 * exactly what it was built for, and it makes the reconnection behaviour
 * something you can check by pressing F5 rather than by reading a test.
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
        <h1 className="text-sm font-medium tracking-wide text-neutral-300">No run here</h1>
        <p className="mt-2 text-[13px] leading-relaxed text-neutral-400">
          The API has no run <span className="font-mono text-neutral-300">{runId}</span>. A run is
          kept for an hour after it finishes, so an older link has expired — or the id is wrong.
        </p>
        <Link
          href="/"
          className="mt-6 self-start rounded border border-neutral-700 px-3 py-1.5 text-[12px] text-neutral-200 hover:border-neutral-500"
        >
          Ask a question
        </Link>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col bg-neutral-950 text-neutral-200 lg:h-screen">
      <header className="flex shrink-0 items-baseline gap-3 border-b border-neutral-800 px-4 py-2">
        <Link href="/" className="text-[13px] font-medium text-neutral-300 hover:text-neutral-100">
          Atlas
        </Link>
        {previous !== null && (
          <Link
            href={`/runs/${previous}`}
            className="shrink-0 text-[12px] text-neutral-500 hover:text-neutral-300"
          >
            ← previous question
          </Link>
        )}
        <p className="truncate text-[12px] text-neutral-400">{state.query ?? 'loading run…'}</p>
        <span className="ml-auto hidden shrink-0 font-mono text-[10px] text-neutral-600 sm:inline">
          {runId}
        </span>
      </header>

      {/* Side by side on wide screens; stacked, and scrolling as one page, on narrow ones. */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="h-[60vh] min-h-[260px] shrink-0 border-b border-neutral-800 lg:h-[52%]">
            {topology === null ? (
              <p className="p-4 text-[12px] text-neutral-500">loading the graph…</p>
            ) : (
              <GraphCanvas topology={topology} state={state} now={now} />
            )}
          </div>

          <div className="min-h-0 flex-1 space-y-6 p-4 lg:overflow-y-auto">
            <Answer state={state} result={result} />
            <Findings state={state} />
            {finished && state.threadId !== null && (
              <section>
                <h2 className="mb-2 text-[11px] uppercase tracking-wide text-neutral-500">
                  follow up
                </h2>
                <QuestionForm
                  threadId={state.threadId}
                  placeholder="Ask a follow-up. It is planned against this answer."
                  hint="continues this conversation"
                  submitLabel="Ask"
                  onStarted={(next) => router.push(`/runs/${next}?from=${runId}`)}
                />
              </section>
            )}
          </div>
        </div>

        <aside className="w-full shrink-0 space-y-3 border-t border-neutral-800 p-3 lg:w-72 lg:overflow-y-auto lg:border-l lg:border-t-0">
          <RunStatus state={state} connection={connection} now={now} />
          <Metrics result={result} />
        </aside>
      </div>
    </main>
  );
}
