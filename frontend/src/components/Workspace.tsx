'use client';

import { useRef, useState } from 'react';
import {
  type ImperativePanelGroupHandle,
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from 'react-resizable-panels';

import { useMediaQuery } from '@/lib/useMediaQuery';

export type PaneId = 'evolution' | 'graph' | 'result';

export interface PaneSpec {
  readonly id: PaneId;
  readonly title: string;
  /** A short line under the title: what the pane is showing right now. */
  readonly subtitle?: string;
  readonly content: React.ReactNode;
}

// Answer-first: the answer is what a reader came for; the timeline and the
// agents explain how it was reached.
const LAYOUT: Readonly<Record<PaneId, number>> = { evolution: 20, graph: 35, result: 45 };
const ORDER: readonly PaneId[] = ['evolution', 'graph', 'result'];

/**
 * The run workspace: evolution, graph, result.
 *
 * Side by side and resizable on a wide screen, the sizes kept per browser; one
 * at a time behind tabs on a narrow one, where three columns would each be too
 * thin to read. Any pane can be maximised and restored.
 */
export function Workspace({
  panes,
  preferredTab = 'graph',
  picked: shownPick,
  onPick,
}: {
  readonly panes: readonly PaneSpec[];
  /**
   * The tab a phone shows until the reader picks one: the agents while the run
   * works, the answer once there is one.
   */
  readonly preferredTab?: PaneId;
  /** The tab picked, when the page keeps it: so an action in one pane can open another. */
  readonly picked?: PaneId | null;
  readonly onPick?: (pane: PaneId) => void;
}): React.JSX.Element {
  const wide = useMediaQuery('(min-width: 1024px)', true);
  const [ownPick, setOwnPick] = useState<PaneId | null>(null);
  const picked = shownPick === undefined ? ownPick : shownPick;
  const setTab = onPick ?? setOwnPick;
  const tab = picked ?? preferredTab;
  const group = useRef<ImperativePanelGroupHandle>(null);
  const [maximised, setMaximised] = useState<PaneId | null>(null);
  const restore = useRef<number[] | null>(null);

  function toggleMaximise(id: PaneId): void {
    const handle = group.current;
    if (handle === null) return;
    if (maximised === id) {
      handle.setLayout(restore.current ?? ORDER.map((pane) => LAYOUT[pane]));
      setMaximised(null);
      return;
    }
    restore.current = handle.getLayout();
    handle.setLayout(ORDER.map((pane) => (pane === id ? 100 : 0)));
    setMaximised(id);
  }

  if (!wide) {
    const active = panes.find((pane) => pane.id === tab) ?? panes[0];
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <div role="tablist" aria-label="Workspace" className="flex shrink-0 border-b border-line bg-surface">
          {panes.map((pane) => (
            <button
              key={pane.id}
              role="tab"
              type="button"
              id={`tab-${pane.id}`}
              aria-selected={pane.id === tab}
              aria-controls={`pane-${pane.id}`}
              onClick={() => setTab(pane.id)}
              className={`flex-1 px-3 py-2 text-xs ${
                pane.id === tab
                  ? 'border-b-2 border-accent font-medium text-ink'
                  : 'text-ink-faint hover:text-ink'
              }`}
            >
              {pane.title}
            </button>
          ))}
        </div>
        {active !== undefined && (
          <div
            id={`pane-${active.id}`}
            role="tabpanel"
            aria-labelledby={`tab-${active.id}`}
            // A flex column, so the pane body gets a height: the graph sizes
            // itself against it, and against nothing it draws nothing.
            className="flex min-h-0 flex-1 flex-col bg-surface"
          >
            <PaneBody pane={active} />
          </div>
        )}
      </div>
    );
  }

  return (
    <PanelGroup
      ref={group}
      direction="horizontal"
      // v2: sizes saved under the old layout would otherwise keep it.
      autoSaveId="atlas-workspace-v2"
      className="min-h-0 flex-1"
    >
      {panes.map((pane, index) => (
        <PanelSlot key={pane.id} index={index}>
          <Panel id={pane.id} order={index + 1} defaultSize={LAYOUT[pane.id]} minSize={0}>
            <section aria-label={pane.title} className="flex h-full min-w-0 flex-col bg-surface">
              <header className="flex shrink-0 items-center gap-2 border-b border-line px-3 py-1.5">
                <div className="min-w-0 flex-1">
                  <h2 className="text-xs font-semibold text-ink">{pane.title}</h2>
                  {pane.subtitle !== undefined && (
                    <p className="truncate text-2xs text-ink-faint">{pane.subtitle}</p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => toggleMaximise(pane.id)}
                  aria-label={maximised === pane.id ? `Restore ${pane.title}` : `Maximise ${pane.title}`}
                  title={maximised === pane.id ? 'Restore' : 'Maximise'}
                  className="rounded p-1 text-ink-faint hover:bg-raised hover:text-ink"
                >
                  <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden="true">
                    <path
                      d={maximised === pane.id ? 'M6 2v4H2M10 14v-4h4' : 'M2 6V2h4M14 10v4h-4'}
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.4"
                      strokeLinecap="round"
                    />
                  </svg>
                </button>
              </header>
              <PaneBody pane={pane} />
            </section>
          </Panel>
        </PanelSlot>
      ))}
    </PanelGroup>
  );
}

/** A resize handle before every pane but the first. */
function PanelSlot({
  index,
  children,
}: {
  readonly index: number;
  readonly children: React.ReactNode;
}): React.JSX.Element {
  return (
    <>
      {index > 0 && (
        <PanelResizeHandle className="w-px bg-line transition-colors hover:bg-accent data-[resize-handle-state=drag]:bg-accent" />
      )}
      {children}
    </>
  );
}

function PaneBody({ pane }: { readonly pane: PaneSpec }): React.JSX.Element {
  return <div className="min-h-0 flex-1 overflow-y-auto">{pane.content}</div>;
}
