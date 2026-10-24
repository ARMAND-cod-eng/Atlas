/**
 * The run's events, as a person reads them: what happened, to which agent, when.
 *
 * A pure function of the run state, so the timeline a live run builds and the
 * one a replay builds are the same by construction. Bookkeeping events (a node
 * starting, the critic's iteration counter) are folded into the entries they
 * belong to rather than listed on their own.
 */

import {
  AGENT_CALL,
  CITATION_CHECK,
  CRITIC_VERDICT,
  FAN_OUT,
  NODE_COMPLETED,
  NODE_FAILED,
  NODE_STARTED,
  REVISION_STARTED,
  RUN_COMPLETED,
  RUN_FAILED,
  RUN_STARTED,
  type AgentEvent,
} from './events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, type RunState } from './reducer';

export type Tone = 'neutral' | 'running' | 'done' | 'failed' | 'retry' | 'edited' | 'law';

export interface TimelineEntry {
  readonly seq: number;
  /** Seconds since the run started. */
  readonly at: number;
  readonly tone: Tone;
  readonly text: string;
  readonly detail: string | null;
  /** The graph node the entry is about, as GraphCanvas ids it, for selection. */
  readonly nodeId: string | null;
}

const STARTED: Readonly<Record<string, string>> = {
  [PLANNER]: 'Planning the research',
  [SYNTHESIZER]: 'Writing the answer',
  [CRITIC]: 'Reviewing the answer against the sources',
};

function nodeId(event: AgentEvent): string | null {
  if (event.node === null) return null;
  const subtask = 'subtask_id' in event.payload ? event.payload.subtask_id : null;
  return event.node === RESEARCHER && subtask ? `${RESEARCHER}:${subtask}` : event.node;
}

function plural(count: number, word: string, many = `${word}s`): string {
  return `${count} ${count === 1 ? word : many}`;
}

function describe(event: AgentEvent): Omit<TimelineEntry, 'seq' | 'at' | 'nodeId'> | null {
  switch (event.type) {
    case RUN_STARTED:
      return { tone: 'neutral', text: 'Question received', detail: event.payload.query };
    case REVISION_STARTED: {
      const carried = event.payload.branches.filter((b) => b.origin === 'carried').length;
      return {
        tone: 'edited',
        text: `Branch ${event.payload.subtask_id} rewritten; ${plural(carried, 'branch', 'branches')} carried over`,
        detail: `${event.payload.before} → ${event.payload.after}`,
      };
    }
    case FAN_OUT:
      return {
        tone: 'neutral',
        text: `Planned ${plural(event.payload.count, 'branch', 'branches')}`,
        detail: null,
      };
    case NODE_STARTED:
      if (event.node === RESEARCHER) return null; // the search that follows says it
      return event.node !== null && STARTED[event.node] !== undefined
        ? { tone: 'running', text: STARTED[event.node] ?? '', detail: null }
        : null;
    case AGENT_CALL: {
      const who = event.payload.subtask_id ?? event.node ?? 'agent';
      if (event.payload.kind === 'tool') {
        const where = event.payload.name === 'library' ? 'the law' : 'the web';
        return {
          tone: event.payload.name === 'library' ? 'law' : 'neutral',
          text: `${who} searched ${where}`,
          detail: event.payload.error,
        };
      }
      const tokens = (event.payload.input_tokens ?? 0) + (event.payload.output_tokens ?? 0);
      return {
        tone: event.payload.error === null ? 'neutral' : 'failed',
        text: `${who} called ${event.payload.name}`,
        detail: event.payload.error ?? (tokens > 0 ? `${tokens.toLocaleString()} tokens` : null),
      };
    }
    case NODE_COMPLETED:
      if (event.node === RESEARCHER) {
        return {
          tone: 'done',
          text: `${event.payload.subtask_id ?? 'branch'} read ${plural(event.payload.sources?.length ?? 0, 'source')}`,
          detail: event.payload.question ?? null,
        };
      }
      return null; // the next step says the last one finished
    case NODE_FAILED:
      return {
        tone: 'failed',
        text: `${event.payload.subtask_id ?? event.node ?? 'a node'} failed`,
        detail: event.payload.error,
      };
    case CRITIC_VERDICT: {
      if (event.payload.overall === 'accept') {
        return { tone: 'done', text: `Review ${event.payload.iteration}: accepted`, detail: null };
      }
      const faulted = event.payload.faulted_subtask_ids;
      const reasons = (event.payload.per_subtask ?? [])
        .filter((v) => v.verdict === 'revise')
        .map((v) => `${v.subtask_id}: ${v.reason}`);
      return {
        tone: 'retry',
        text:
          faulted.length > 0
            ? `Review ${event.payload.iteration}: sent ${faulted.join(', ')} back`
            : `Review ${event.payload.iteration}: the write-up needs work`,
        detail:
          [...reasons, event.payload.synthesis_feedback ?? ''].filter(Boolean).join('\n') || null,
      };
    }
    case CITATION_CHECK: {
      if (event.payload.ok) {
        return {
          tone: 'law',
          text: `${plural(event.payload.cited.length, 'citation')} checked against the law`,
          detail: null,
        };
      }
      const problems = [
        ...event.payload.unresolved.map((label) => `${label} does not exist`),
        ...event.payload.unretrieved.map((label) => `${label} was never retrieved`),
        ...event.payload.misquoted.map(([label]) => `a quotation of ${label} is not its words`),
        ...event.payload.ungrounded_urls.map((url) => `${url} was never retrieved`),
      ];
      return {
        tone: 'failed',
        text: `Citation check: ${plural(problems.length, 'problem')}`,
        detail: problems.join('\n'),
      };
    }
    case RUN_COMPLETED:
      return {
        tone: event.payload.degraded.length > 0 ? 'retry' : 'done',
        text: 'Answer ready',
        detail: event.payload.degraded.length > 0 ? event.payload.degraded.join(', ') : null,
      };
    case RUN_FAILED:
      return { tone: 'failed', text: 'Run failed', detail: event.payload.error };
    default:
      return null;
  }
}

export function timelineEntries(state: RunState): readonly TimelineEntry[] {
  const start = state.startedAt === null ? null : Date.parse(state.startedAt);
  const entries: TimelineEntry[] = [];
  for (const event of state.timeline) {
    const described = describe(event);
    if (described === null) continue;
    entries.push({
      ...described,
      seq: event.seq,
      at: start === null ? 0 : Math.max(0, (Date.parse(event.ts) - start) / 1000),
      nodeId: nodeId(event),
    });
  }
  return entries;
}
