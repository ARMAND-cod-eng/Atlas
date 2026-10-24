'use client';

import { useState } from 'react';

import { Answer } from './Answer';
import { Findings } from './Findings';
import { GraphCanvas } from './GraphCanvas';
import { Metrics } from './Metrics';
import { RunStatus } from './RunStatus';
import { Timeline } from './Timeline';
import { type PaneSpec, Workspace } from './Workspace';
import type { RunResult, Topology } from '@/lib/events';
import type { RunState } from '@/lib/reducer';
import type { Connection } from '@/lib/useRunStream';

function evolutionLine(state: RunState): string {
  const reviews = state.criticRounds.length;
  const parts = [`${state.timeline.length} events`];
  if (reviews > 0) parts.push(`${reviews} review${reviews === 1 ? '' : 's'}`);
  if (state.revision !== null) parts.push(`revises ${state.revision.subtaskId}`);
  return parts.join(' · ');
}

function agentsLine(state: RunState): string {
  if (!state.fannedOut) return 'planning';
  const carried = state.branches.filter((b) => b.origin === 'carried').length;
  const line = `${state.branches.length} branch${state.branches.length === 1 ? '' : 'es'}`;
  return carried > 0 ? `${line}, ${carried} carried over` : line;
}

/**
 * One run, in the three-pane workspace: how it evolved, the agents doing the
 * work, and what they produced. Shared by the live run page and the demo replay,
 * so both are the same screen fed differently.
 */
export function RunWorkspace({
  state,
  result,
  topology,
  now,
  connection,
  followUp = null,
}: {
  readonly state: RunState;
  readonly result: RunResult | null;
  readonly topology: Topology | null;
  readonly now: number;
  readonly connection: Connection;
  readonly followUp?: React.ReactNode;
}): React.JSX.Element {
  const [selected, setSelected] = useState<string | null>(null);

  const panes: PaneSpec[] = [
    {
      id: 'evolution',
      title: 'Evolution',
      subtitle: evolutionLine(state),
      content: (
        <>
          <div className="border-b border-line p-3">
            <RunStatus state={state} connection={connection} now={now} />
          </div>
          <Timeline state={state} selected={selected} onSelect={setSelected} />
        </>
      ),
    },
    {
      id: 'graph',
      title: 'Agents',
      subtitle: agentsLine(state),
      content: (
        <div className="flex h-full min-h-[320px] flex-col">
          <div className="min-h-[240px] flex-1">
            {topology === null ? (
              <p className="p-4 text-xs text-ink-faint">loading the graph…</p>
            ) : (
              <GraphCanvas topology={topology} state={state} now={now} />
            )}
          </div>
          <div className="h-28 shrink-0 border-t border-line bg-canvas p-3">
            <p className="text-xs text-ink-faint">
              {selected === null
                ? 'Select an agent, here or in the timeline, to see its task, the exact prompt it was sent and what it produced.'
                : `Selected: ${selected}`}
            </p>
          </div>
        </div>
      ),
    },
    {
      id: 'result',
      title: 'Result',
      subtitle: state.answer === null ? 'not written yet' : 'answer ready',
      content: (
        <div className="space-y-6 p-4">
          <Answer state={state} result={result} />
          <Findings state={state} />
          {followUp}
          <Metrics result={result} />
        </div>
      ),
    },
  ];

  return <Workspace panes={panes} />;
}
