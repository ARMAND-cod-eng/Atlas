'use client';

import { formatDegraded } from '@/lib/format';
import type { RunState } from '@/lib/reducer';
import { type StepStatus, runSteps } from '@/lib/timeline';

const MARK: Readonly<Record<StepStatus, { readonly dot: string; readonly label: string }>> = {
  idle: { dot: 'border border-line-strong', label: 'not started' },
  running: { dot: 'bg-running animate-pulse', label: 'in progress' },
  done: { dot: 'bg-done', label: 'done' },
  failed: { dot: 'bg-failed', label: 'failed' },
};

/**
 * Where the run has got to, as the four steps every run takes, and anything
 * that went wrong on the way, said prominently: a degraded run that looked like
 * a clean one is the failure this project keeps designing against.
 */
export function RunProgress({ state }: { readonly state: RunState }): React.JSX.Element {
  const steps = runSteps(state);
  return (
    // A container, so a step's detail goes under it when the pane is too narrow
    // for both on one line, rather than being cut off.
    <section aria-label="Progress" className="@container border-b border-line p-3">
      <ol className="space-y-1.5">
        {steps.map((step) => (
          <li key={step.key} className="flex flex-wrap items-center gap-x-2.5 text-xs">
            <span
              role="img"
              aria-label={MARK[step.status].label}
              className={`h-2.5 w-2.5 shrink-0 rounded-full ${MARK[step.status].dot}`}
            />
            <span className={step.status === 'idle' ? 'text-ink-faint' : 'font-medium text-ink'}>{step.label}</span>
            {step.detail !== null && (
              <span
                className={`w-full pl-5 text-2xs @[15rem]:ml-auto @[15rem]:w-auto @[15rem]:pl-0 ${
                  step.status === 'failed' ? 'text-failed' : 'text-ink-muted'
                }`}
              >
                {step.detail}
              </span>
            )}
          </li>
        ))}
      </ol>
      <RunBanners state={state} />
    </section>
  );
}

function RunBanners({ state }: { readonly state: RunState }): React.JSX.Element | null {
  const notes = [
    ...state.degraded.map((flag) => ({ tone: 'retry' as const, text: formatDegraded(flag) })),
    ...(state.degradedStream
      ? [
          {
            tone: 'retry' as const,
            text: 'Stream incomplete: an event was lost and could not be recovered, so this view may be missing a step.',
          },
        ]
      : []),
    ...(state.error !== null ? [{ tone: 'failed' as const, text: state.error }] : []),
  ];
  if (notes.length === 0) return null;
  return (
    <ul className="mt-3 space-y-1.5">
      {notes.map((note) => (
        <li
          key={note.text}
          className={`rounded-md border px-2.5 py-1.5 text-2xs ${
            note.tone === 'failed' ? 'border-failed/40 bg-failed/10 text-failed' : 'border-retry/40 bg-retry/10 text-retry'
          }`}
        >
          {note.text}
        </li>
      ))}
    </ul>
  );
}
