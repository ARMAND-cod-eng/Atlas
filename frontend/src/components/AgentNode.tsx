'use client';

import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';

import { elapsedSeconds, durationSeconds, formatSeconds } from '@/lib/format';
import type { NodeStatus } from '@/lib/reducer';

export interface AgentNodeData extends Record<string, unknown> {
  readonly label: string;
  readonly caption: string | null;
  readonly status: NodeStatus;
  readonly startedAt: string | null;
  readonly finishedAt: string | null;
  readonly error: string | null;
  /** > 1 means the critic sent this branch back. */
  readonly runs: number;
  readonly sourceCount: number | null;
  /** A branch a person rewrote, or one carried over from the version revised. */
  readonly origin: 'planned' | 'edited' | 'carried' | null;
  readonly now: number;
}

export type AgentNodeType = Node<AgentNodeData, 'agent'>;

/**
 * Status is carried by colour *and* by a word.
 *
 * Colour alone fails anyone who cannot distinguish the hues, and it fails
 * screenshots. The label is the thing a viewer can actually verify.
 */
const STATUS_STYLES: Record<NodeStatus, { ring: string; dot: string; text: string }> = {
  idle: { ring: 'border-line', dot: 'bg-idle', text: 'text-ink-faint' },
  running: { ring: 'border-running/70', dot: 'bg-running animate-pulse', text: 'text-running' },
  done: { ring: 'border-done/60', dot: 'bg-done', text: 'text-done' },
  failed: { ring: 'border-failed/70', dot: 'bg-failed', text: 'text-failed' },
};

export function AgentNode({ data, selected }: NodeProps<AgentNodeType>): React.JSX.Element {
  const style = STATUS_STYLES[data.status];
  const elapsed =
    data.status === 'running'
      ? elapsedSeconds(data.startedAt, data.now)
      : durationSeconds(data.startedAt, data.finishedAt);

  return (
    <div
      className={`w-[210px] cursor-pointer rounded-md border bg-canvas/90 px-3 py-2 ${style.ring} ${
        selected ? 'ring-2 ring-accent ring-offset-1 ring-offset-canvas' : ''
      } ${data.origin === 'carried' ? 'opacity-70' : ''}`}
      data-testid={`node-${data.label}`}
      data-status={data.status}
    >
      <Handle type="target" position={Position.Left} className="!bg-line-strong" />

      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 shrink-0 rounded-full ${style.dot}`} />
        <span className="truncate text-sm font-medium text-ink">{data.label}</span>
        {data.origin === 'edited' && (
          <span className="ml-auto shrink-0 rounded bg-edited/15 px-1.5 py-0.5 text-2xs font-medium text-edited">
            edited
          </span>
        )}
        {data.origin === 'carried' && (
          <span className="ml-auto shrink-0 rounded bg-raised px-1.5 py-0.5 text-2xs text-ink-muted">
            carried over
          </span>
        )}
        {data.runs > 1 && (
          // The visible marker of a targeted retry, on the branch that earned it.
          <span className="ml-auto shrink-0 rounded bg-retry/15 px-1.5 py-0.5 text-2xs font-medium text-retry">
            run {data.runs}
          </span>
        )}
      </div>

      {data.caption !== null && (
        <p className="mt-1 line-clamp-2 text-2xs leading-snug text-ink-muted">{data.caption}</p>
      )}

      <div className="mt-1.5 flex items-center gap-2 text-2xs tabular-nums">
        <span className={style.text}>{data.status}</span>
        {elapsed !== null && <span className="text-ink-faint">{formatSeconds(elapsed)}</span>}
        {data.sourceCount !== null && data.sourceCount > 0 && (
          <span className="text-ink-faint">{data.sourceCount} sources</span>
        )}
      </div>

      {data.error !== null && (
        <p className="mt-1 truncate text-2xs text-failed" title={data.error}>
          {data.error}
        </p>
      )}

      <Handle type="source" position={Position.Right} className="!bg-line-strong" />
    </div>
  );
}
