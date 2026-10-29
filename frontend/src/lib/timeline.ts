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
  CITATION_SUPPORT,
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
  /** The event it describes. */
  readonly type: AgentEvent['type'];
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

function describe(event: AgentEvent): Omit<TimelineEntry, 'seq' | 'at' | 'nodeId' | 'type'> | null {
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
    case CITATION_SUPPORT: {
      const { supported, unsupported, unverified, unchecked } = event.payload;
      const notes = [
        unverified.length > 0 ? `${plural(unverified.length, 'citation')} claimed supported without the provision's words` : '',
        unchecked > 0 ? `${plural(unchecked, 'citation')} not judged` : '',
      ].filter(Boolean);
      if (unsupported.length === 0) {
        return {
          tone: 'law',
          text: `Support check: ${plural(supported.length, 'citation')} ${supported.length === 1 ? 'says what it is' : 'say what they are'} cited for`,
          detail: notes.join('\n') || null,
        };
      }
      return {
        tone: 'failed',
        text: `Support check: ${plural(unsupported.length, 'citation')} ${unsupported.length === 1 ? 'does not say what it is' : 'do not say what they are'} cited for`,
        detail: [...unsupported.map((entry) => `${entry.label}: ${entry.reason}`), ...notes].join('\n'),
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
      type: event.type,
    });
  }
  return entries;
}

// --- the run as steps ------------------------------------------------------------------

export type StepKey = 'edit' | 'plan' | 'research' | 'write' | 'review' | 'end';
export type StepStatus = 'idle' | 'running' | 'done' | 'failed';

export interface Step {
  readonly key: 'plan' | 'research' | 'write' | 'review';
  readonly label: string;
  readonly status: StepStatus;
  /** What the step came to, in a few words; null when there is nothing yet. */
  readonly detail: string | null;
}

/** The four steps every run takes, and how far it has got through them. */
export function runSteps(state: RunState): readonly Step[] {
  const branches = state.branches;
  const failedBranches = branches.filter((b) => b.status === 'failed').length;
  const revising = state.revision !== null;

  const plan: Step = revising
    ? { key: 'plan', label: 'Plan', status: 'done', detail: 'carried over' }
    : {
        key: 'plan',
        label: 'Plan',
        status: state.nodes[PLANNER]?.status ?? 'idle',
        detail: state.fannedOut ? plural(branches.length, 'branch', 'branches') : null,
      };

  let researchStatus: StepStatus = 'idle';
  if (state.fannedOut || revising) {
    if (branches.some((b) => b.status === 'running' || b.status === 'idle')) researchStatus = 'running';
    else if (branches.length > 0 && failedBranches === branches.length) researchStatus = 'failed';
    else researchStatus = 'done';
  }
  const carried = branches.filter((b) => b.origin === 'carried').length;
  const read = branches.reduce((total, b) => total + b.sources.length, 0);
  const research: Step = {
    key: 'research',
    label: 'Research',
    status: researchStatus,
    detail:
      failedBranches > 0
        ? `${failedBranches} of ${branches.length} failed`
        : revising
          ? `${branches.length - carried} rewritten, ${carried} carried over`
          : read > 0
            ? `${plural(read, 'source')} read`
            : null,
  };

  const drafts = state.nodes[SYNTHESIZER]?.runs ?? 0;
  const write: Step = {
    key: 'write',
    label: 'Write',
    status: state.nodes[SYNTHESIZER]?.status ?? 'idle',
    detail: drafts > 1 ? `${drafts} drafts` : null,
  };

  const rounds = state.criticRounds;
  const last = rounds.at(-1);
  const review: Step = {
    key: 'review',
    label: 'Review',
    status: state.nodes[CRITIC]?.status ?? 'idle',
    detail:
      last === undefined
        ? null
        : last.overall === 'accept'
          ? `accepted in round ${last.iteration}`
          : state.phase === 'completed'
            ? `not accepted after ${plural(rounds.length, 'round')}`
            : `round ${last.iteration}: sent back`,
  };
  return [plan, research, write, review];
}

const STEP_LABEL: Readonly<Record<StepKey, string>> = {
  edit: 'Your edit',
  plan: 'Plan',
  research: 'Research',
  write: 'Write',
  review: 'Review',
  end: 'Finished',
};

function stepOf(entry: TimelineEntry, revising: boolean): StepKey {
  switch (entry.type) {
    case RUN_STARTED:
      return revising ? 'edit' : 'plan';
    case REVISION_STARTED:
      return 'edit';
    case FAN_OUT:
      return 'plan';
    case RUN_COMPLETED:
    case RUN_FAILED:
      return 'end';
    case CITATION_CHECK:
      return 'write';
    case CITATION_SUPPORT:
    case CRITIC_VERDICT:
      return 'review';
    default:
      break;
  }
  if (entry.nodeId === PLANNER) return 'plan';
  if (entry.nodeId?.startsWith(`${RESEARCHER}:`) === true || entry.nodeId === RESEARCHER) return 'research';
  if (entry.nodeId === SYNTHESIZER) return 'write';
  if (entry.nodeId === CRITIC) return 'review';
  return 'end';
}

export interface TimelineGroup {
  readonly key: StepKey;
  readonly round: number;
  readonly label: string;
  readonly entries: readonly TimelineEntry[];
}

/**
 * The timeline cut into the steps it went through, in the order they happened:
 * a branch the critic sent back opens "Research · round 2" rather than joining
 * the first research. Internal calls (each search, each model call) are left
 * out unless asked for: the Attempts tab lists them all.
 */
export function groupedTimeline(
  state: RunState,
  { showCalls }: { readonly showCalls: boolean },
): readonly TimelineGroup[] {
  const revising = state.revision !== null;
  const groups: { key: StepKey; round: number; label: string; entries: TimelineEntry[] }[] = [];
  let round = 1;
  for (const entry of timelineEntries(state)) {
    if (!showCalls && entry.type === AGENT_CALL) continue;
    const key = stepOf(entry, revising);
    const current = groups.at(-1);
    if (current === undefined || current.key !== key) {
      const counted = key === 'research' || key === 'write' || key === 'review';
      groups.push({
        key,
        round: counted ? round : 1,
        label: counted && round > 1 ? `${STEP_LABEL[key]} · round ${round}` : STEP_LABEL[key],
        entries: [entry],
      });
    } else {
      current.entries.push(entry);
    }
    // A verdict closes a round: what follows it belongs to the next.
    if (entry.type === CRITIC_VERDICT) round += 1;
  }
  return groups;
}
