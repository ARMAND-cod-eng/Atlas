/**
 * The event contract, mirroring `atlas/events.py`.
 *
 * Payloads are modelled per event type rather than as one loose bag, so the
 * reducer cannot read `subtask_id` off a `run_completed` and quietly get
 * `undefined`. The discriminated union means TypeScript narrows the payload from
 * the `type` field, and a backend payload change surfaces here as a type error
 * instead of as a blank field on screen.
 */

export const RUN_STARTED = 'run_started';
export const NODE_STARTED = 'node_started';
export const NODE_COMPLETED = 'node_completed';
export const NODE_FAILED = 'node_failed';
export const FAN_OUT = 'fan_out';
export const ITERATION_STARTED = 'iteration_started';
export const CRITIC_VERDICT = 'critic_verdict';
export const RUN_COMPLETED = 'run_completed';
export const RUN_FAILED = 'run_failed';

/** A stream ends deliberately with one of these, or it broke. */
export const TERMINAL_TYPES = [RUN_COMPLETED, RUN_FAILED] as const;

export type TerminalType = (typeof TERMINAL_TYPES)[number];

export function isTerminal(type: string): type is TerminalType {
  return (TERMINAL_TYPES as readonly string[]).includes(type);
}

/** A source a researcher actually retrieved. */
export interface SourceRef {
  readonly url: string;
  readonly title: string;
}

interface Base<T extends string, P> {
  readonly run_id: string;
  readonly seq: number;
  readonly ts: string;
  readonly type: T;
  readonly node: string | null;
  readonly payload: P;
}

export type AgentEvent =
  | Base<typeof RUN_STARTED, { readonly query: string; readonly thread_id: string }>
  | Base<typeof NODE_STARTED, { readonly subtask_id: string | null }>
  | Base<
      typeof NODE_COMPLETED,
      {
        readonly subtask_id: string | null;
        // Present on researcher events only — the sources that branch found.
        readonly sources?: readonly SourceRef[];
        readonly question?: string;
      }
    >
  | Base<
      typeof NODE_FAILED,
      {
        readonly subtask_id: string | null;
        readonly error: string;
        readonly sources?: readonly SourceRef[];
        readonly question?: string;
      }
    >
  | Base<
      typeof FAN_OUT,
      {
        readonly subtask_ids: readonly string[];
        readonly count: number;
        readonly questions?: Readonly<Record<string, string>>;
      }
    >
  | Base<typeof ITERATION_STARTED, { readonly iteration: number }>
  | Base<
      typeof CRITIC_VERDICT,
      {
        readonly iteration: number;
        readonly overall: 'accept' | 'revise';
        readonly faulted_subtask_ids: readonly string[];
        readonly has_synthesis_feedback: boolean;
      }
    >
  | Base<
      typeof RUN_COMPLETED,
      {
        readonly answer: string;
        readonly thread_id: string;
        readonly iterations: number;
        readonly subtask_count: number;
        readonly source_count: number;
        readonly degraded: readonly string[];
        readonly total_tokens: number;
      }
    >
  | Base<typeof RUN_FAILED, { readonly error: string; readonly error_type?: string }>;

export type EventType = AgentEvent['type'];

/** The topology served by `GET /graph/topology`. */
export interface TopologyNode {
  readonly id: string;
  readonly fan_out: boolean;
  readonly terminal: boolean;
}

export interface TopologyEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly conditional: boolean;
}

export interface Topology {
  readonly nodes: readonly TopologyNode[];
  readonly edges: readonly TopologyEdge[];
}

/** The full result from `GET /runs/{id}`. */
export interface RunResult {
  readonly answer: string;
  readonly thread_id: string;
  readonly iterations: number;
  readonly subtasks: readonly { readonly id: string; readonly question: string }[];
  readonly sources: readonly { readonly url: string; readonly title: string; readonly snippet: string }[];
  readonly metrics: {
    readonly degraded: readonly string[];
    readonly node_timings: readonly { readonly node: string; readonly duration_s: number }[];
    readonly llm_calls: readonly {
      readonly node: string;
      readonly provider: string;
      readonly model: string;
      readonly input_tokens: number | null;
      readonly output_tokens: number | null;
      readonly duration_s: number;
    }[];
    readonly critic_rounds: readonly {
      readonly iteration: number;
      readonly overall: string;
      readonly faulted_subtask_ids: readonly string[];
    }[];
  };
}
