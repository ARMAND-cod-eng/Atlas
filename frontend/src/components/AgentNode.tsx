'use client';

import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';

import { Badge } from './ui';
import { elapsedSeconds, durationSeconds, formatSeconds } from '@/lib/format';
import { NODE_WIDTH } from '@/lib/layout';
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
const STATUS_STYLES: Record<NodeStatus, { ring: string; dot: string; text: string; word: string }> = {
  idle: { ring: 'border-line', dot: 'bg-idle', text: 'text-ink-faint', word: 'waiting' },
  running: { ring: 'border-running/70', dot: 'bg-running animate-pulse', text: 'text-running', word: 'running' },
  done: { ring: 'border-line-strong', dot: 'bg-done', text: 'text-done', word: 'done' },
  failed: { ring: 'border-failed/70', dot: 'bg-failed', text: 'text-failed', word: 'failed' },
};

export function AgentNode({ data, selected }: NodeProps<AgentNodeType>): React.JSX.Element {
  const style = STATUS_STYLES[data.status];
  const elapsed =
    data.status === 'running'
      ? elapsedSeconds(data.startedAt, data.now)
      : durationSeconds(data.startedAt, data.finishedAt);

  return (
    <div
      style={{ width: NODE_WIDTH }}
      className={`cursor-pointer rounded-lg border bg-surface px-3 py-2 shadow-sm transition-shadow hover:shadow-md ${style.ring} ${
        selected ? 'ring-2 ring-accent ring-offset-2 ring-offset-canvas' : ''
      } ${data.origin === 'carried' ? 'opacity-70' : ''}`}
      data-testid={`node-${data.label}`}
      data-status={data.status}
    >
      <Handle type="target" position={Position.Top} className="!border-0 !bg-line-strong" />

      <div className="flex items-center gap-2">
        <span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-full ${style.dot}`} />
        <span className="min-w-0 truncate text-sm font-semibold text-ink">{data.label}</span>
        <span className="ml-auto flex shrink-0 gap-1">
          {data.origin === 'edited' && <Badge tone="edited">edited</Badge>}
          {data.origin === 'carried' && <Badge>carried over</Badge>}
          {/* The visible marker of a targeted retry, on the branch that earned it. */}
          {data.runs > 1 && <Badge tone="retry">run {data.runs}</Badge>}
        </span>
      </div>

      {data.caption !== null && (
        <p className="mt-0.5 truncate text-2xs text-ink-muted" title={data.caption}>
          {data.caption}
        </p>
      )}

      <p className="mt-1 flex items-center gap-1.5 text-2xs tabular-nums">
        <span className={style.text}>{style.word}</span>
        {elapsed !== null && <span className="text-ink-faint">· {formatSeconds(elapsed)}</span>}
        {data.sourceCount !== null && data.sourceCount > 0 && (
          <span className="text-ink-faint">· {data.sourceCount} sources</span>
        )}
      </p>

      {data.error !== null && (
        <p className="mt-1 truncate text-2xs text-failed" title={data.error}>
          {data.error}
        </p>
      )}

      <Handle type="source" position={Position.Bottom} className="!border-0 !bg-line-strong" />
    </div>
  );
}
