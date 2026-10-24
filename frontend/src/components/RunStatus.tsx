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
    <section className="rounded-md border border-line bg-canvas p-3">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-medium text-ink">{label}</h2>
        <span className="text-sm tabular-nums text-ink-muted">{formatSeconds(runElapsed)}</span>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-y-1.5 text-2xs">
        <dt className="text-ink-faint">iteration</dt>
        <dd className="text-right tabular-nums text-ink">{state.iteration || '—'}</dd>

        <dt className="text-ink-faint">tokens</dt>
        <dd className="text-right tabular-nums text-ink">{formatTokens(state.totalTokens)}</dd>

        <dt className="text-ink-faint">sources</dt>
        <dd className="text-right tabular-nums text-ink">{sourceCount}</dd>

        <dt className="text-ink-faint">stream</dt>
        <dd className="text-right text-ink">{connection}</dd>
      </dl>

      {state.criticRounds.length > 0 && (
        <div className="mt-3 border-t border-line pt-2">
          <p className="text-2xs uppercase tracking-wide text-ink-faint">critic</p>
          <ul className="mt-1 space-y-1">
            {state.criticRounds.map((round) => (
              <li key={round.iteration} className="flex items-baseline gap-2 text-2xs">
                <span className="tabular-nums text-ink-faint">#{round.iteration}</span>
                <span className={round.overall === 'accept' ? 'text-done' : 'text-retry'}>
                  {round.overall}
                </span>
                {round.faultedSubtaskIds.length > 0 && (
                  <span className="text-ink-muted">
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
              className="rounded border border-retry/40 bg-retry/10 px-2 py-1 text-2xs text-retry"
            >
              {formatDegraded(flag)}
            </li>
          ))}
        </ul>
      )}

      {state.degradedStream && (
        <p className="mt-3 rounded border border-retry/40 bg-retry/10 px-2 py-1 text-2xs text-retry">
          stream incomplete — an event was lost and could not be recovered, so this view may be
          missing a step
        </p>
      )}

      {state.error !== null && (
        <p className="mt-3 rounded border border-failed/40 bg-failed/10 px-2 py-1 text-2xs text-failed">
          {state.error}
        </p>
      )}
    </section>
  );
}
