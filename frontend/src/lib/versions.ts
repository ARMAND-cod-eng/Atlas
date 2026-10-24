/**
 * The runs on one thread, as versions of an answer: the first question, each
 * follow-up, and each revision of a branch, numbered in the order they began.
 *
 * And what changed between two of them, word by word.
 */

import { diffWords } from 'diff';

import type { RunVersion } from './api';

export interface VersionRow {
  readonly runId: string;
  /** v1, v2…: the order the runs began in. */
  readonly number: number;
  readonly kind: 'question' | 'follow-up' | 'revision';
  readonly title: string;
  /** The question asked, or the branch's rewritten question. */
  readonly detail: string;
  /** What a revised branch asked before. */
  readonly before: string | null;
  /** The version this one continues or revises. */
  readonly basedOn: number | null;
  readonly status: RunVersion['status'];
  readonly current: boolean;
}

export function versionRows(runs: readonly RunVersion[], currentRunId: string): readonly VersionRow[] {
  const numbers = new Map(runs.map((run, index) => [run.run_id, index + 1]));
  return runs.map((run, index) => {
    const basedOn = run.parent_run_id === null ? null : (numbers.get(run.parent_run_id) ?? null);
    const common = {
      runId: run.run_id,
      number: index + 1,
      basedOn,
      status: run.status,
      current: run.run_id === currentRunId,
    };
    if (run.kind === 'revision' && run.revision !== null) {
      return {
        ...common,
        kind: 'revision' as const,
        title: `Edited ${run.revision.subtask_id}`,
        detail: run.revision.after,
        before: run.revision.before,
      };
    }
    // Any question after the thread's first continues the conversation.
    const followUp = index > 0;
    return {
      ...common,
      kind: followUp ? ('follow-up' as const) : ('question' as const),
      title: followUp ? 'Follow-up' : 'Question',
      detail: run.query,
      before: null,
    };
  });
}

/**
 * The versions worth comparing this one with: every other finished one, the
 * one it is based on first.
 */
export function comparable(rows: readonly VersionRow[]): readonly VersionRow[] {
  const current = rows.find((row) => row.current);
  const others = rows.filter((row) => !row.current && row.status === 'completed');
  return [
    ...others.filter((row) => row.number === current?.basedOn),
    ...others.filter((row) => row.number !== current?.basedOn),
  ];
}

export interface DiffPart {
  readonly kind: 'same' | 'added' | 'removed';
  readonly text: string;
}

function words(text: string): number {
  return text.split(/\s+/).filter((word) => word !== '').length;
}

/** What changed from one answer to the other, and how many words either way. */
export function answerDiff(
  before: string,
  after: string,
): { readonly parts: readonly DiffPart[]; readonly added: number; readonly removed: number } {
  const parts: DiffPart[] = diffWords(before, after).map((change) => ({
    kind: change.added ? 'added' : change.removed ? 'removed' : 'same',
    text: change.value,
  }));
  return {
    parts,
    added: parts.filter((part) => part.kind === 'added').reduce((sum, part) => sum + words(part.text), 0),
    removed: parts.filter((part) => part.kind === 'removed').reduce((sum, part) => sum + words(part.text), 0),
  };
}
