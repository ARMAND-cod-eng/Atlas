'use client';

import Link from 'next/link';
import type { ReactNode } from 'react';

import { ThemeToggle } from './ThemeToggle';
import { Badge, buttonClass } from './ui';
import type { RunState } from '@/lib/reducer';
import { runStatus } from '@/lib/runStatus';

/**
 * The run page's header: where you are (the question, as the page's title),
 * how the run is doing, and the way to ask something else.
 */
export function RunHeader({
  state,
  now,
  badges,
  actions,
  breadcrumb,
}: {
  readonly state: RunState;
  readonly now: number;
  /** Marks on this run: a replay, scripted models. */
  readonly badges?: ReactNode;
  readonly actions?: ReactNode;
  /** A quiet line under the title: where this run came from. */
  readonly breadcrumb?: ReactNode;
}): React.JSX.Element {
  const status = runStatus(state, now);
  return (
    <header className="shrink-0 border-b border-line bg-surface">
      <div className="flex items-center gap-3 px-4 py-2.5">
        <Link href="/" className="shrink-0 text-sm font-semibold text-ink hover:text-accent">
          Atlas
        </Link>
        <span aria-hidden="true" className="h-4 w-px shrink-0 bg-line" />
        <h1 className="min-w-0 flex-1 truncate text-base font-medium text-ink" title={state.query ?? undefined}>
          {state.query ?? 'Loading the run…'}
        </h1>
        {badges}
        <Badge tone={status.tone}>
          {status.tone === 'running' && (
            <span aria-hidden="true" className="h-1.5 w-1.5 animate-pulse rounded-full bg-running" />
          )}
          {status.text}
        </Badge>
        {actions}
        <Link href="/" className={`${buttonClass('secondary', 'sm')} hidden sm:inline-flex`}>
          New question
        </Link>
        <ThemeToggle />
      </div>
      {breadcrumb !== undefined && <div className="-mt-1 px-4 pb-2 text-2xs text-ink-faint">{breadcrumb}</div>}
    </header>
  );
}
