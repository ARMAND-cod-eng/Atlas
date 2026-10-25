'use client';

import Link from 'next/link';

import { RunWorkspace } from '@/components/RunWorkspace';
import { ThemeToggle } from '@/components/ThemeToggle';
import calls from '@/demo/calls.json';
import compliance from '@/demo/compliance.json';
import topology from '@/demo/topology.json';
import type { AgentCall } from '@/lib/api';
import type { AgentEvent, Topology } from '@/lib/events';
import { useReplay } from '@/lib/useReplay';

// Recorded by backend/scripts/capture_fixtures.py from the real graph and
// runner, with the models faked: the shapes are the backend's own.
const EVENTS = compliance as readonly AgentEvent[];
const TOPOLOGY = topology as Topology;

// Every call the run made, in full, as a live run's API would serve it: the
// inspector shows the demo's prompts with no backend behind it.
const CALLS = new Map((calls as readonly AgentCall[]).map((call) => [call.call_id, call]));
const recordedCall = (callId: string): Promise<AgentCall | null> => Promise.resolve(CALLS.get(callId) ?? null);

/**
 * A recorded compliance run, replayed through the same reducer and workspace a
 * live run uses, with no backend: researchers reading the regulation library,
 * a citation check catching an invented article, and the rewrite that fixes it.
 */
export default function DemoPage(): React.JSX.Element {
  const { state, done, restart } = useReplay(EVENTS);
  const last = state.timeline.at(-1);
  // The recording's own clock, so elapsed times read as they did in the run.
  const now = last === undefined ? 0 : Date.parse(last.ts);

  return (
    <main className="flex h-screen flex-col bg-canvas text-ink">
      <header className="flex shrink-0 items-center gap-3 border-b border-line bg-surface px-4 py-2">
        <Link href="/" className="text-sm font-medium text-ink hover:text-accent">
          Atlas
        </Link>
        <span className="rounded bg-raised px-1.5 py-0.5 text-2xs text-ink-muted">demo replay</span>
        <p className="min-w-0 flex-1 truncate text-xs text-ink-muted">{state.query ?? 'starting…'}</p>
        <button
          type="button"
          onClick={restart}
          disabled={!done}
          className="rounded border border-line px-2 py-0.5 text-2xs text-ink-muted hover:border-ink-faint hover:text-ink disabled:opacity-40"
        >
          Replay
        </button>
        <ThemeToggle />
      </header>
      <RunWorkspace
        runId={null}
        loadCall={recordedCall}
        state={state}
        result={null}
        topology={TOPOLOGY}
        now={now}
        connection={done ? 'closed' : 'open'}
      />
    </main>
  );
}
