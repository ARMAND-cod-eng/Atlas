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
  idle: { ring: 'border-neutral-800', dot: 'bg-neutral-600', text: 'text-neutral-500' },
  running: { ring: 'border-sky-500/70', dot: 'bg-sky-400 animate-pulse', text: 'text-sky-300' },
  done: { ring: 'border-emerald-600/60', dot: 'bg-emerald-500', text: 'text-emerald-400' },
  failed: { ring: 'border-red-600/70', dot: 'bg-red-500', text: 'text-red-400' },
};

export function AgentNode({ data }: NodeProps<AgentNodeType>): React.JSX.Element {
  const style = STATUS_STYLES[data.status];
  const elapsed =
    data.status === 'running'
      ? elapsedSeconds(data.startedAt, data.now)
      : durationSeconds(data.startedAt, data.finishedAt);

  return (
    <div
      className={`w-[210px] rounded-md border bg-neutral-950/90 px-3 py-2 ${style.ring}`}
      data-testid={`node-${data.label}`}
      data-status={data.status}
    >
      <Handle type="target" position={Position.Left} className="!bg-neutral-700" />

      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 shrink-0 rounded-full ${style.dot}`} />
        <span className="truncate text-[13px] font-medium text-neutral-100">{data.label}</span>
        {data.runs > 1 && (
          // The visible marker of a targeted retry, on the branch that earned it.
          <span className="ml-auto shrink-0 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
            run {data.runs}
          </span>
        )}
      </div>

      {data.caption !== null && (
        <p className="mt-1 line-clamp-2 text-[11px] leading-snug text-neutral-400">{data.caption}</p>
      )}

      <div className="mt-1.5 flex items-center gap-2 text-[10px] tabular-nums">
        <span className={style.text}>{data.status}</span>
        {elapsed !== null && <span className="text-neutral-500">{formatSeconds(elapsed)}</span>}
        {data.sourceCount !== null && data.sourceCount > 0 && (
          <span className="text-neutral-500">{data.sourceCount} sources</span>
        )}
      </div>

      {data.error !== null && (
        <p className="mt-1 truncate text-[10px] text-red-400" title={data.error}>
          {data.error}
        </p>
      )}

      <Handle type="source" position={Position.Right} className="!bg-neutral-700" />
    </div>
  );
}
