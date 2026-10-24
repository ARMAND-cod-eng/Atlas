'use client';

import { useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

import { Answer } from './Answer';
import { Findings } from './Findings';
import { GraphCanvas } from './GraphCanvas';
import { Inspector } from './Inspector';
import { Metrics } from './Metrics';
import { RunStatus } from './RunStatus';
import { Timeline } from './Timeline';
import { Compare, Versions } from './Versions';
import { type PaneSpec, Workspace } from './Workspace';
import type { RunResult, Topology } from '@/lib/events';
import type { RunState } from '@/lib/reducer';
import type { Connection } from '@/lib/useRunStream';
import { useVersions } from '@/lib/useVersions';
import { versionRows } from '@/lib/versions';

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
  runId,
  state,
  result,
  topology,
  now,
  connection,
  followUp = null,
  onRevised,
}: {
  /** The run's id, or null for a replay: nothing behind it to fetch or revise. */
  readonly runId: string | null;
  readonly state: RunState;
  readonly result: RunResult | null;
  readonly topology: Topology | null;
  readonly now: number;
  readonly connection: Connection;
  readonly followUp?: React.ReactNode;
  readonly onRevised?: (runId: string) => void;
}): React.JSX.Element {
  const [selected, setSelected] = useState<string | null>(null);
  // Asked again when the run's phase changes: finishing changes its own row.
  const runs = useVersions(state.threadId, runId, state.phase);
  const rows = runs === null || runId === null ? [] : versionRows(runs, runId);

  const panes: PaneSpec[] = [
    {
      id: 'evolution',
      title: 'Evolution',
      subtitle: evolutionLine(state),
      content: (
        <>
          <Versions rows={rows} />
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
        <PanelGroup direction="vertical" autoSaveId="atlas-agents" className="min-h-[480px]">
          <Panel defaultSize={55} minSize={25}>
            {topology === null ? (
              <p className="p-4 text-xs text-ink-faint">loading the graph…</p>
            ) : (
              <GraphCanvas topology={topology} state={state} now={now} selected={selected} onSelect={setSelected} />
            )}
          </Panel>
          <PanelResizeHandle className="h-px bg-line transition-colors hover:bg-accent data-[resize-handle-state=drag]:bg-accent" />
          <Panel defaultSize={45} minSize={15} className="bg-canvas">
            <Inspector runId={runId} state={state} result={result} selected={selected} onRevised={onRevised} />
          </Panel>
        </PanelGroup>
      ),
    },
    {
      id: 'result',
      title: 'Result',
      subtitle: state.answer === null ? 'not written yet' : 'answer ready',
      content: (
        <div className="space-y-6 p-4">
          <Compare rows={rows} answer={state.answer} />
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
