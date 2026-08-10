'use client';

import { durationSeconds, elapsedSeconds, formatDegraded, formatSeconds, formatTokens } from '@/lib/format';
import { CRITIC, SYNTHESIZER, type RunState } from '@/lib/reducer';
import type { Connection } from '@/lib/useRunStream';

interface Props {
  readonly state: RunState;
  readonly connection: Connection;
  readonly now: number;
}

/**
 * What is happening right now, and for how long.
 *
 * This panel is the answer to the long silent stretch. When the synthesizer is
 * working there are no events for 20–45 seconds, and two things are true: it
 * started, and it started *n* seconds ago. Both are shown. Nothing else is,
 * because nothing else is known — a progress bar here would be an invention.
 */
export function RunStatus({ state, connection, now }: Props): React.JSX.Element {
  const active = Object.values(state.nodes).find((node) => node.status === 'running');
  const activeBranches = state.branches.filter((branch) => branch.status === 'running');
  // A finished run's duration comes from its own timestamps, so it stops moving
  // and stays identical on a replayed view — a wall-clock reading would drift
  // every time the page was reopened.
  const runElapsed =
    state.finishedAt !== null
      ? durationSeconds(state.startedAt, state.finishedAt)
      : elapsedSeconds(state.startedAt, now);

  // Before the run ends there is no authoritative count, so the branches are
  // summed; afterwards the terminal event's deduplicated figure wins.
  const retrieved = state.branches.reduce((total, branch) => total + branch.sources.length, 0);
  const sourceCount = state.sourceCount ?? (retrieved === 0 ? '—' : retrieved);

  const label =
    state.phase === 'completed'
      ? 'complete'
      : state.phase === 'failed'
        ? 'failed'
        : activeBranches.length > 0
          ? `researching · ${activeBranches.length} branch${activeBranches.length === 1 ? '' : 'es'}`
          : active?.id === SYNTHESIZER
            ? 'synthesising'
            : active?.id === CRITIC
              ? 'reviewing'
              : active !== undefined
                ? active.id
                : 'starting';

  return (
    <section className="rounded-md border border-neutral-800 bg-neutral-950 p-3">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-[13px] font-medium text-neutral-200">{label}</h2>
        <span className="text-[13px] tabular-nums text-neutral-400">{formatSeconds(runElapsed)}</span>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-y-1.5 text-[11px]">
        <dt className="text-neutral-500">iteration</dt>
        <dd className="text-right tabular-nums text-neutral-300">{state.iteration || '—'}</dd>

        <dt className="text-neutral-500">tokens</dt>
        <dd className="text-right tabular-nums text-neutral-300">{formatTokens(state.totalTokens)}</dd>

        <dt className="text-neutral-500">sources</dt>
        <dd className="text-right tabular-nums text-neutral-300">{sourceCount}</dd>

        <dt className="text-neutral-500">stream</dt>
        <dd className="text-right text-neutral-300">{connection}</dd>
      </dl>

      {state.criticRounds.length > 0 && (
        <div className="mt-3 border-t border-neutral-800 pt-2">
          <p className="text-[10px] uppercase tracking-wide text-neutral-500">critic</p>
          <ul className="mt-1 space-y-1">
            {state.criticRounds.map((round) => (
              <li key={round.iteration} className="flex items-baseline gap-2 text-[11px]">
                <span className="tabular-nums text-neutral-500">#{round.iteration}</span>
                <span className={round.overall === 'accept' ? 'text-emerald-400' : 'text-amber-400'}>
                  {round.overall}
                </span>
                {round.faultedSubtaskIds.length > 0 && (
                  <span className="text-neutral-400">
                    sent back {round.faultedSubtaskIds.join(', ')}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {state.degraded.length > 0 && (
        // Prominent, not tucked away. A degraded run that looks identical to a
        // clean one is the failure this project keeps designing against.
        <ul className="mt-3 space-y-1">
          {state.degraded.map((flag) => (
            <li
              key={flag}
              className="rounded border border-amber-700/40 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-300"
            >
              {formatDegraded(flag)}
            </li>
          ))}
        </ul>
      )}

      {state.degradedStream && (
        <p className="mt-3 rounded border border-amber-700/40 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-300">
          stream incomplete — an event was lost and could not be recovered, so this view may be
          missing a step
        </p>
      )}

      {state.error !== null && (
        <p className="mt-3 rounded border border-red-700/40 bg-red-500/10 px-2 py-1 text-[11px] text-red-300">
          {state.error}
        </p>
      )}
    </section>
  );
}
