'use client';

import { Badge, Button, Disclosure } from './ui';
import { hostOf } from '@/lib/format';
import type { RunState } from '@/lib/reducer';

interface Props {
  readonly state: RunState;
  /** Open while there is no answer: then it is the only thing to read. */
  readonly open?: boolean;
  /** Rewrite one branch and re-run it; absent where the run cannot be revised. */
  readonly onEdit?: (nodeId: string) => void;
}

/**
 * The sources each branch retrieved, as they land.
 *
 * This is the other half of the answer to the silent stretch. By the time the
 * synthesizer starts, every researcher has finished and its sources exist — the
 * event stream simply used to throw them away and report a count at the end.
 * Showing them is not progress indication; it is work already done that was
 * previously hidden, and it gives the user something real to read for the
 * 20–45 seconds during which nothing else can honestly be said.
 *
 * Once there is an answer it is also where a branch is redirected from: the
 * question that came back with the wrong sources is right there, beside them.
 */
export function Findings({ state, open = false, onEdit }: Props): React.JSX.Element | null {
  if (state.branches.length === 0) return null;

  return (
    <Disclosure title="Research by branch" count={state.branches.length} defaultOpen={open}>
      <ul className="divide-y divide-line">
        {state.branches.map((branch) => (
          <li key={branch.id} className="py-3 first:pt-0 last:pb-0">
            <div className="flex items-start gap-2">
              <span className="shrink-0 pt-px font-mono text-2xs text-ink-faint">{branch.subtaskId}</span>
              <p className="min-w-0 flex-1 text-xs font-medium leading-snug text-ink">{branch.question}</p>
              {branch.origin === 'edited' && <Badge tone="edited">edited</Badge>}
              {branch.origin === 'carried' && <Badge>carried over</Badge>}
              {onEdit !== undefined && (
                <Button
                  variant="ghost"
                  size="sm"
                  aria-label={`Edit & re-run ${branch.subtaskId}`}
                  onClick={() => onEdit(branch.id)}
                  className="-my-1"
                >
                  Edit &amp; re-run
                </Button>
              )}
            </div>

            {branch.status === 'failed' ? (
              <p className="mt-1 pl-7 text-2xs text-failed">
                {branch.error ?? 'this branch failed'} — the answer does not include it
              </p>
            ) : branch.sources.length === 0 ? (
              <p className="mt-1 pl-7 text-2xs text-ink-faint">
                {branch.status === 'running' ? 'searching…' : 'no sources'}
              </p>
            ) : (
              <ul className="mt-1.5 space-y-0.5 pl-7">
                {branch.sources.map((source, index) => (
                  // Keyed by position too: two paragraphs of one article share its URL.
                  <li key={`${source.label ?? source.url}:${index}`} className="truncate text-2xs">
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-ink-muted hover:text-accent"
                      title={source.title}
                    >
                      {source.label !== null && source.label !== undefined && (
                        <span className="mr-1.5 font-medium text-law">{source.label}</span>
                      )}
                      {source.title}
                    </a>
                    {(source.label === null || source.label === undefined) && (
                      <span className="ml-1.5 text-ink-faint">{hostOf(source.url)}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </Disclosure>
  );
}
