/**
 * Events in, run state out. Pure, synchronous, and the only place in this app
 * where anything non-obvious happens — which is why it is the only place with
 * real tests.
 *
 * Two properties it has to guarantee, both of which come from Phase 3's `seq`
 * being dense:
 *
 * **Ordering is handled, not assumed.** Events are applied strictly in `seq`
 * order. Anything ahead of the next expected `seq` waits in a buffer and drains
 * when the hole fills. Out-of-order arrival and a transient gap are the same
 * situation seen at different moments, so they share one mechanism rather than
 * two that can disagree.
 *
 * **A gap is never silently applied.** Skipping ahead would render a state the
 * run never passed through — a lane stuck `running` forever because its
 * `node_completed` was the event that went missing. The buffer makes the hole
 * explicit so the caller can refetch.
 *
 * Replay-driven state must be *identical* to live-driven state, because
 * reconnection replays. Nothing here reads a clock or the network; timestamps
 * come off the events themselves.
 */

import {
  CRITIC_VERDICT,
  FAN_OUT,
  ITERATION_STARTED,
  NODE_COMPLETED,
  NODE_FAILED,
  NODE_STARTED,
  RUN_COMPLETED,
  RUN_FAILED,
  RUN_STARTED,
  type AgentEvent,
  type SourceRef,
  isTerminal,
} from './events';

export type NodeStatus = 'idle' | 'running' | 'done' | 'failed';

export interface NodeState {
  readonly id: string;
  readonly status: NodeStatus;
  /** ISO timestamp of `node_started`, so elapsed time is derivable. */
  readonly startedAt: string | null;
  readonly finishedAt: string | null;
  readonly error: string | null;
  /** How many times this node has run. > 1 means the critic sent it back. */
  readonly runs: number;
}

export interface BranchState extends NodeState {
  readonly subtaskId: string;
  readonly question: string;
  readonly sources: readonly SourceRef[];
}

export interface CriticRound {
  readonly iteration: number;
  readonly overall: 'accept' | 'revise';
  readonly faultedSubtaskIds: readonly string[];
}

export type RunPhase = 'connecting' | 'running' | 'completed' | 'failed';

export interface RunState {
  readonly phase: RunPhase;
  readonly query: string | null;
  readonly threadId: string | null;
  readonly startedAt: string | null;
  readonly finishedAt: string | null;
  /** Skeleton nodes, keyed by node id. Researchers live in `branches`. */
  readonly nodes: Readonly<Record<string, NodeState>>;
  /** One per subtask, once `fan_out` has told us how many there are. */
  readonly branches: readonly BranchState[];
  readonly fannedOut: boolean;
  readonly iteration: number;
  readonly criticRounds: readonly CriticRound[];
  /** Subtask ids the most recent verdict sent back. Drives the retry edge. */
  readonly retryTargets: readonly string[];
  readonly retryFired: boolean;
  readonly answer: string | null;
  readonly error: string | null;
  readonly degraded: readonly string[];
  readonly totalTokens: number | null;
  readonly sourceCount: number | null;

  // --- stream bookkeeping ---
  /** The next `seq` we can apply. Events at or below this are duplicates. */
  readonly nextSeq: number;
  /** Events that arrived ahead of the hole, keyed by seq. */
  readonly buffered: Readonly<Record<number, AgentEvent>>;
  /** The seq we are missing, if the buffer is stuck. */
  readonly gapAt: number | null;
  /** Set when recovery has been tried and the same hole is still there. */
  readonly degradedStream: boolean;
}

/** Skeleton node ids, matching `atlas/graph/nodes.py`. */
export const PLANNER = 'planner';
export const RESEARCHER = 'researcher';
export const SYNTHESIZER = 'synthesizer';
export const CRITIC = 'critic';
/** Where a person's edit to one branch enters a finished run. Never runs itself. */
export const HUMAN_EDIT = 'human_edit';

function idleNode(id: string): NodeState {
  return { id, status: 'idle', startedAt: null, finishedAt: null, error: null, runs: 0 };
}

export function initialState(nodeIds: readonly string[]): RunState {
  return {
    phase: 'connecting',
    query: null,
    threadId: null,
    startedAt: null,
    finishedAt: null,
    nodes: Object.fromEntries(nodeIds.map((id) => [id, idleNode(id)])),
    branches: [],
    fannedOut: false,
    iteration: 0,
    criticRounds: [],
    retryTargets: [],
    retryFired: false,
    answer: null,
    error: null,
    degraded: [],
    totalTokens: null,
    sourceCount: null,
    nextSeq: 1,
    buffered: {},
    gapAt: null,
    degradedStream: false,
  };
}

function withNode(
  state: RunState,
  id: string,
  change: (node: NodeState) => NodeState,
): Readonly<Record<string, NodeState>> {
  const current = state.nodes[id] ?? idleNode(id);
  return { ...state.nodes, [id]: change(current) };
}

function withBranch(
  branches: readonly BranchState[],
  subtaskId: string,
  change: (branch: BranchState) => BranchState,
): readonly BranchState[] {
  return branches.map((branch) => (branch.subtaskId === subtaskId ? change(branch) : branch));
}

/**
 * Apply one event, assuming it is the next in sequence.
 *
 * Split out from {@link accept} so ordering and interpretation stay separable:
 * this function is about what an event *means*, and knows nothing about gaps.
 */
function apply(state: RunState, event: AgentEvent): RunState {
  switch (event.type) {
    case RUN_STARTED:
      return {
        ...state,
        phase: 'running',
        query: event.payload.query,
        threadId: event.payload.thread_id,
        startedAt: event.ts,
      };

    case FAN_OUT: {
      const questions = event.payload.questions ?? {};
      // The researcher placeholder is replaced by one lane per subtask. This is
      // the only topology change in a normal run, and the only point at which
      // layout is allowed to recompute.
      return {
        ...state,
        fannedOut: true,
        branches: event.payload.subtask_ids.map((subtaskId) => ({
          ...idleNode(`${RESEARCHER}:${subtaskId}`),
          subtaskId,
          question: questions[subtaskId] ?? subtaskId,
          sources: [],
        })),
      };
    }

    case NODE_STARTED: {
      const subtaskId = event.payload.subtask_id;
      if (subtaskId !== null) {
        return {
          ...state,
          branches: withBranch(state.branches, subtaskId, (branch) => ({
            ...branch,
            status: 'running',
            startedAt: event.ts,
            finishedAt: null,
            error: null,
            runs: branch.runs + 1,
          })),
        };
      }
      if (event.node === null) return state;
      return {
        ...state,
        nodes: withNode(state, event.node, (node) => ({
          ...node,
          status: 'running',
          startedAt: event.ts,
          finishedAt: null,
          error: null,
          runs: node.runs + 1,
        })),
      };
    }

    case NODE_COMPLETED: {
      const subtaskId = event.payload.subtask_id;
      if (subtaskId !== null) {
        return {
          ...state,
          branches: withBranch(state.branches, subtaskId, (branch) => ({
            ...branch,
            status: 'done',
            finishedAt: event.ts,
            question: event.payload.question ?? branch.question,
            sources: event.payload.sources ?? branch.sources,
          })),
        };
      }
      if (event.node === null) return state;
      return {
        ...state,
        nodes: withNode(state, event.node, (node) => ({
          ...node,
          status: 'done',
          finishedAt: event.ts,
        })),
      };
    }

    case NODE_FAILED: {
      const subtaskId = event.payload.subtask_id;
      if (subtaskId !== null) {
        // A branch that failed while its siblings succeeded. Phase 1 keeps the
        // run alive by turning the failure into data, so the run will still
        // complete with an answer — the UI must show that the answer rests on
        // four fifths of the research rather than presenting a clean green run.
        return {
          ...state,
          branches: withBranch(state.branches, subtaskId, (branch) => ({
            ...branch,
            status: 'failed',
            finishedAt: event.ts,
            error: event.payload.error,
            question: event.payload.question ?? branch.question,
            sources: event.payload.sources ?? branch.sources,
          })),
        };
      }
      if (event.node === null) return state;
      return {
        ...state,
        nodes: withNode(state, event.node, (node) => ({
          ...node,
          status: 'failed',
          finishedAt: event.ts,
          error: event.payload.error,
        })),
      };
    }

    case ITERATION_STARTED:
      return { ...state, iteration: event.payload.iteration };

    case CRITIC_VERDICT: {
      const faulted = event.payload.faulted_subtask_ids;
      return {
        ...state,
        criticRounds: [
          ...state.criticRounds,
          {
            iteration: event.payload.iteration,
            overall: event.payload.overall,
            faultedSubtaskIds: faulted,
          },
        ],
        retryTargets: faulted,
        // Latches. The retry edge should stay legible after the fact, not blink
        // out the moment the branches finish re-running.
        retryFired: state.retryFired || faulted.length > 0,
      };
    }

    case RUN_COMPLETED:
      return {
        ...state,
        phase: 'completed',
        finishedAt: event.ts,
        answer: event.payload.answer,
        iteration: event.payload.iterations,
        degraded: event.payload.degraded,
        totalTokens: event.payload.total_tokens,
        sourceCount: event.payload.source_count,
        // A run can end while a node is mid-flight only if something went wrong;
        // leaving a spinner running forever would be the visible symptom.
        nodes: settle(state.nodes),
      };

    case RUN_FAILED:
      return {
        ...state,
        phase: 'failed',
        finishedAt: event.ts,
        error: event.payload.error,
        nodes: settle(state.nodes),
      };
  }
}

/** Anything still `running` when the run ended never finished. */
function settle(nodes: Readonly<Record<string, NodeState>>): Readonly<Record<string, NodeState>> {
  return Object.fromEntries(
    Object.entries(nodes).map(([id, node]) =>
      node.status === 'running' ? [id, { ...node, status: 'idle' satisfies NodeStatus }] : [id, node],
    ),
  );
}

/**
 * Feed one event in, respecting sequence.
 *
 * Duplicates (`seq < nextSeq`) are dropped — replay after a reconnect overlaps
 * by design, and applying `node_started` twice would inflate the run counter
 * that distinguishes a retried branch from a fresh one.
 */
export function accept(state: RunState, event: AgentEvent): RunState {
  if (event.seq < state.nextSeq) return state;

  if (event.seq > state.nextSeq) {
    return {
      ...state,
      buffered: { ...state.buffered, [event.seq]: event },
      gapAt: state.nextSeq,
    };
  }

  let next = { ...apply(state, event), nextSeq: state.nextSeq + 1 };

  // Drain anything that was waiting on this one.
  const buffered = { ...state.buffered };
  for (;;) {
    const pending = buffered[next.nextSeq];
    if (pending === undefined) break;
    delete buffered[next.nextSeq];
    next = { ...apply(next, pending), nextSeq: next.nextSeq + 1 };
  }

  return {
    ...next,
    buffered,
    gapAt: Object.keys(buffered).length > 0 ? next.nextSeq : null,
  };
}

export function acceptAll(state: RunState, events: readonly AgentEvent[]): RunState {
  return events.reduce(accept, state);
}

/** The last contiguous seq — where a reconnect should resume from. */
export function resumeFrom(state: RunState): number {
  return state.nextSeq - 1;
}

/** True once the stream has delivered a terminal event. */
export function isFinished(state: RunState): boolean {
  return state.phase === 'completed' || state.phase === 'failed';
}

export function terminalTypeOf(event: AgentEvent): boolean {
  return isTerminal(event.type);
}
