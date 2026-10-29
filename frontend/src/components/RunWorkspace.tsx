'use client';

import { useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

import { AnswerCard, Sources } from './Answer';
import { Findings } from './Findings';
import { GraphCanvas } from './GraphCanvas';
import { type CallSource, Inspector } from './Inspector';
import { Metrics } from './Metrics';
import { RunDetails } from './RunDetails';
import { RunProgress } from './RunProgress';
import { Timeline } from './Timeline';
import { Versions } from './Versions';
import { Disclaimer } from './Disclaimer';
import { type PaneId, type PaneSpec, Workspace } from './Workspace';
import { Disclosure } from './ui';
import type { RunResult, Topology } from '@/lib/events';
import { type RunState, isFinished } from '@/lib/reducer';
import { useInspectorLink } from '@/lib/inspectorLink';
import { agentOrder, useShortcuts } from '@/lib/shortcuts';
import type { Connection } from '@/lib/useRunStream';
import { useVersions } from '@/lib/useVersions';
import { versionRows } from '@/lib/versions';

function evolutionLine(state: RunState): string {
  const reviews = state.criticRounds.length;
  const parts: string[] = [];
  if (state.revision !== null) parts.push(`revises ${state.revision.subtaskId}`);
  if (state.fannedOut) parts.push(`${state.branches.length} branch${state.branches.length === 1 ? '' : 'es'}`);
  if (reviews > 0) parts.push(`${reviews} review${reviews === 1 ? '' : 's'}`);
  return parts.length > 0 ? parts.join(' · ') : 'planning';
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
  loadCall,
  state,
  result,
  topology,
  now,
  connection,
  followUp = null,
  onRevised,
  onResearchAnyway,
  onAskGeneral,
}: {
  /** The run's id, or null for a replay: nothing behind it to fetch or revise. */
  readonly runId: string | null;
  /** A call's full record, for the inspector's prompts: the API's, or a recording's. */
  readonly loadCall: CallSource | null;
  readonly state: RunState;
  readonly result: RunResult | null;
  readonly topology: Topology | null;
  readonly now: number;
  readonly connection: Connection;
  readonly followUp?: React.ReactNode;
  readonly onRevised?: (runId: string) => void;
  /** Research a question judged out of scope after all; resolves to why it could not, or null. */
  readonly onResearchAnyway?: () => Promise<string | null>;
  /** Ask a question judged out of scope as a general one; same contract. */
  readonly onAskGeneral?: () => Promise<string | null>;
}): React.JSX.Element {
  // In the URL, so a link can open one agent's prompt and a reload keeps it.
  const [link, setSelected, setTab] = useInspectorLink();
  const selected = link.node;
  // The pane a phone shows, kept here so an action in one pane can open another.
  const [pane, setPane] = useState<PaneId | null>(null);

  /** To a branch's question, where the run can be revised: from its key, or from the answer. */
  function editBranch(nodeId: string): void {
    const branch = state.branches.find((b) => b.id === nodeId);
    if (branch === undefined) return;
    setSelected(nodeId);
    setTab('task');
    // On a phone the question is in another tab.
    setPane('graph');
    window.setTimeout(() => document.getElementById(`edit-${branch.subtaskId}`)?.focus(), 0);
  }
  const editable = runId !== null && isFinished(state) && state.phase === 'completed';

  useShortcuts({
    order: agentOrder(state),
    selected,
    onSelect: setSelected,
    // e: to the selected branch's question.
    onEdit: () => {
      if (selected !== null) editBranch(selected);
    },
  });
  // Asked again when the run's phase changes: finishing changes its own row.
  const runs = useVersions(state.threadId, runId, state.phase);
  const rows = runs === null || runId === null ? [] : versionRows(runs, runId);

  const panes: PaneSpec[] = [
    {
      id: 'evolution',
      title: 'Timeline',
      subtitle: evolutionLine(state),
      content: (
        <>
          <Versions rows={rows} />
          <RunProgress state={state} />
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
            <Inspector
              runId={runId}
              loadCall={loadCall}
              state={state}
              result={result}
              selected={selected}
              tab={link.tab}
              onTab={setTab}
              onRevised={onRevised}
            />
          </Panel>
        </PanelGroup>
      ),
    },
    {
      id: 'result',
      title: 'Answer',
      subtitle:
        state.declined !== null
          ? 'not researched'
          : state.phase === 'failed'
          ? 'the run failed'
          : state.answer === null
            ? 'being written'
            : 'ready',
      content: (
        // Answer first; then what supports it, folded; then what to do next.
        <div className="mx-auto max-w-3xl space-y-3 p-4 sm:p-5">
          <AnswerCard
            state={state}
            result={result}
            rows={rows}
            onResearchAnyway={onResearchAnyway}
            onAskGeneral={onAskGeneral}
          />
          <Sources result={result} />
          <Findings state={state} open={state.answer === null} onEdit={editable ? editBranch : undefined} />
          {followUp}
          <Disclosure title="Run details">
            <div className="space-y-4">
              <RunDetails runId={runId} state={state} connection={connection} />
              <Metrics result={result} />
            </div>
          </Disclosure>
          <Disclaimer className="pt-3" />
        </div>
      ),
    },
  ];

  return (
    <Workspace
      panes={panes}
      preferredTab={state.answer === null ? 'graph' : 'result'}
      picked={pane}
      onPick={setPane}
    />
  );
}
