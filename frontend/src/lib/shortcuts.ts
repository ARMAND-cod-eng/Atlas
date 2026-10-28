'use client';

import { useEffect } from 'react';

import type { RunState } from './reducer';
import { timelineEntries } from './timeline';

/**
 * The workspace's keys, for reading a run without the mouse:
 *
 * - j / k: the next or previous agent, in the order they first ran;
 * - e: edit the selected branch's question (where the run allows it);
 * - Escape: close the inspector.
 *
 * Never while someone is typing, and never with a modifier held: ctrl+k and
 * the like belong to the browser.
 */

/** Every agent the run reached, in the order it first did something. */
export function agentOrder(state: RunState): readonly string[] {
  const seen = new Set<string>();
  for (const entry of timelineEntries(state)) {
    if (entry.nodeId !== null) seen.add(entry.nodeId);
  }
  return [...seen];
}

/** The agent `delta` steps from the selected one, stopping at either end. */
export function stepAgent(order: readonly string[], selected: string | null, delta: 1 | -1): string | null {
  if (order.length === 0) return null;
  const at = selected === null ? -1 : order.indexOf(selected);
  if (at === -1) return delta === 1 ? (order[0] ?? null) : (order.at(-1) ?? null);
  return order[Math.min(Math.max(at + delta, 0), order.length - 1)] ?? null;
}

function isTyping(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))
  );
}

export function useShortcuts({
  order,
  selected,
  onSelect,
  onEdit,
}: {
  readonly order: readonly string[];
  readonly selected: string | null;
  readonly onSelect: (node: string | null) => void;
  readonly onEdit: () => void;
}): void {
  useEffect(() => {
    function onKey(event: KeyboardEvent): void {
      if (event.ctrlKey || event.metaKey || event.altKey || isTyping(event.target)) return;
      if (event.key === 'j' || event.key === 'k') {
        event.preventDefault();
        onSelect(stepAgent(order, selected, event.key === 'j' ? 1 : -1));
      } else if (event.key === 'e') {
        event.preventDefault();
        onEdit();
      } else if (event.key === 'Escape') {
        onSelect(null);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [order, selected, onSelect, onEdit]);
}
