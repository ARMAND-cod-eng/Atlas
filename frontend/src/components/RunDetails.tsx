'use client';

import { formatTokens } from '@/lib/format';
import type { RunState } from '@/lib/reducer';
import type { Connection } from '@/lib/useRunStream';

/**
 * The run's bookkeeping, for whoever needs it: its id, tokens, sources,
 * review rounds and the stream's state. Kept out of the timeline, where it
 * was noise to anyone reading the answer.
 */
export function RunDetails({
  runId,
  state,
  connection,
}: {
  readonly runId: string | null;
  readonly state: RunState;
  readonly connection: Connection;
}): React.JSX.Element {
  const retrieved = state.branches.reduce((total, branch) => total + branch.sources.length, 0);
  const rows: readonly (readonly [string, string])[] = [
    ['Run', runId ?? 'a recording'],
    ['Tokens', formatTokens(state.totalTokens)],
    ['Sources', String(state.sourceCount ?? (retrieved === 0 ? '—' : retrieved))],
    ['Review rounds', String(state.criticRounds.length || '—')],
    ['Stream', connection],
  ];
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-2xs">
      {rows.map(([name, value]) => (
        <div key={name} className="contents">
          <dt className="text-ink-muted">{name}</dt>
          <dd className="truncate text-right font-mono text-ink">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
