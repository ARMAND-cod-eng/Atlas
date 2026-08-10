'use client';

import Link from 'next/link';
import { use } from 'react';

import { Answer } from '@/components/Answer';
import { Findings } from '@/components/Findings';
import { GraphCanvas } from '@/components/GraphCanvas';
import { Metrics } from '@/components/Metrics';
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
 */
export default function RunPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}): React.JSX.Element {
  const { runId } = use(params);
  const topology = useTopology();
  const { state, connection, result } = useRunStream(runId);
  const now = useNow(!isFinished(state));

  return (
    <main className="flex h-screen flex-col bg-neutral-950 text-neutral-200">
      <header className="flex shrink-0 items-baseline gap-3 border-b border-neutral-800 px-4 py-2">
        <Link href="/" className="text-[13px] font-medium text-neutral-300 hover:text-neutral-100">
          Atlas
        </Link>
        <p className="truncate text-[12px] text-neutral-400">{state.query ?? 'loading run…'}</p>
        <span className="ml-auto shrink-0 font-mono text-[10px] text-neutral-600">{runId}</span>
      </header>

      <div className="flex min-h-0 flex-1">
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="h-[52%] min-h-[260px] shrink-0 border-b border-neutral-800">
            {topology === null ? (
              <p className="p-4 text-[12px] text-neutral-500">loading the graph…</p>
            ) : (
              <GraphCanvas topology={topology} state={state} now={now} />
            )}
          </div>

          <div className="min-h-0 flex-1 space-y-6 overflow-y-auto p-4">
            <Answer state={state} result={result} />
            <Findings state={state} />
          </div>
        </div>

        <aside className="w-72 shrink-0 space-y-3 overflow-y-auto border-l border-neutral-800 p-3">
          <RunStatus state={state} connection={connection} now={now} />
          <Metrics result={result} />
        </aside>
      </div>
    </main>
  );
}
