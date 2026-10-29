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
/** A compliance question judged to be about neither act: the run ends at the plan. */
export const SCOPE_DECLINED = 'scope_declined';
export const ITERATION_STARTED = 'iteration_started';
export const CRITIC_VERDICT = 'critic_verdict';
export const RUN_COMPLETED = 'run_completed';
export const RUN_FAILED = 'run_failed';
/** One call an agent made, to a model or a search: a summary; the full call is fetched. */
export const AGENT_CALL = 'agent_call';
/** A run that revises another: which branch was edited, which were carried over. */
export const REVISION_STARTED = 'revision_started';
/** A compliance draft's citations, checked against the law. */
export const CITATION_CHECK = 'citation_check';
/** The same draft's citations, judged by a model for whether each says what it is cited for. */
export const CITATION_SUPPORT = 'citation_support';

/** A stream ends deliberately with one of these, or it broke. */
export const TERMINAL_TYPES = [RUN_COMPLETED, RUN_FAILED] as const;

/** What a run researches: the AI Act and the GDPR, its citations checked; or anything, from the web. */
export type Domain = 'compliance' | 'general';

export type TerminalType = (typeof TERMINAL_TYPES)[number];

export function isTerminal(type: string): type is TerminalType {
  return (TERMINAL_TYPES as readonly string[]).includes(type);
}

/** How much weight a source carries. Only primary law says what the law requires. */
export type Authority = 'primary_law' | 'official_guidance' | 'secondary';

/** A source a researcher actually retrieved. */
export interface SourceRef {
  readonly url: string;
  readonly title: string;
  /** A provision's citation label, "AI Act Art. 6(2)". Absent for a web page. */
  readonly label?: string | null;
  readonly authority?: Authority;
}

/** One citation, judged against its provision: the quote that supports it, or why not. */
export interface SupportEntry {
  readonly label: string;
  readonly sentence: string;
  readonly quote: string;
  readonly reason: string;
}

/** One agent call as the live stream carries it. */
export interface CallSummary {
  readonly call_id: string;
  readonly span_id: string | null;
  readonly subtask_id: string | null;
  readonly kind: 'model' | 'tool';
  /** The model that answered, or the tool: "library", "web". */
  readonly name: string;
  readonly duration_s: number;
  readonly input_tokens: number | null;
  readonly output_tokens: number | null;
  readonly error: string | null;
  readonly preview: string;
}

/** The critic's judgement on one branch. */
export interface SubtaskVerdict {
  readonly subtask_id: string;
  readonly verdict: 'accept' | 'revise';
  readonly reason: string;
  readonly guidance: string;
}

/** A branch as a revision starts: edited, or carried over from the parent. */
export interface RevisionBranch {
  readonly subtask_id: string;
  readonly question: string;
  readonly origin: 'carried' | 'edited';
  readonly sources: readonly SourceRef[];
  readonly error: string | null;
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
  | Base<
      typeof RUN_STARTED,
      {
        readonly query: string;
        readonly thread_id: string;
        /** Absent from streams recorded before runs carried it. */
        readonly domain?: Domain;
      }
    >
  | Base<
      typeof NODE_STARTED,
      { readonly subtask_id: string | null; readonly span_id?: string }
    >
  | Base<
      typeof NODE_COMPLETED,
      {
        readonly subtask_id: string | null;
        readonly span_id?: string;
        // Present on researcher events only — the sources that branch found.
        readonly sources?: readonly SourceRef[];
        readonly question?: string;
        readonly findings_preview?: string;
      }
    >
  | Base<
      typeof NODE_FAILED,
      {
        readonly subtask_id: string | null;
        readonly span_id?: string;
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
  | Base<typeof SCOPE_DECLINED, { readonly reason: string }>
  | Base<typeof ITERATION_STARTED, { readonly iteration: number }>
  | Base<
      typeof CRITIC_VERDICT,
      {
        readonly iteration: number;
        readonly overall: 'accept' | 'revise';
        readonly faulted_subtask_ids: readonly string[];
        readonly has_synthesis_feedback: boolean;
        readonly per_subtask?: readonly SubtaskVerdict[];
        readonly synthesis_feedback?: string | null;
      }
    >
  | Base<typeof AGENT_CALL, CallSummary>
  | Base<
      typeof REVISION_STARTED,
      {
        readonly parent_run_id: string;
        readonly subtask_id: string;
        readonly before: string;
        readonly after: string;
        readonly branches: readonly RevisionBranch[];
      }
    >
  | Base<
      typeof CITATION_CHECK,
      {
        readonly ok: boolean;
        readonly cited: readonly string[];
        readonly unresolved: readonly string[];
        readonly unretrieved: readonly string[];
        /** [the provision cited, the quotation it does not contain] */
        readonly misquoted: readonly (readonly [string, string])[];
        readonly ungrounded_urls: readonly string[];
      }
    >
  | Base<
      typeof CITATION_SUPPORT,
      {
        /** The critic round, which is the draft it judged. */
        readonly iteration: number;
        readonly supported: readonly SupportEntry[];
        readonly unsupported: readonly SupportEntry[];
        /** Judged supported, but the quoted support is not in the provision's text. */
        readonly unverified: readonly SupportEntry[];
        readonly unchecked: number;
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

/**
 * Every event type, for the stream's listeners. `EventSource` needs one listener
 * per named type, so a type left out here is one that never arrives — which is
 * how the three types added with revisions and the citation check nearly went
 * missing. The check below makes leaving one out a type error.
 */
export const EVENT_TYPES = [
  RUN_STARTED,
  NODE_STARTED,
  NODE_COMPLETED,
  NODE_FAILED,
  FAN_OUT,
  SCOPE_DECLINED,
  ITERATION_STARTED,
  CRITIC_VERDICT,
  AGENT_CALL,
  REVISION_STARTED,
  CITATION_CHECK,
  CITATION_SUPPORT,
  RUN_COMPLETED,
  RUN_FAILED,
] as const;

type Unlisted = Exclude<EventType, (typeof EVENT_TYPES)[number]>;
/** Fails to compile, naming the type, if an event type is missing from EVENT_TYPES. */
export const EVERY_EVENT_TYPE_IS_LISTED: [Unlisted] extends [never] ? true : Unlisted = true;

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
  readonly sources: readonly {
    readonly url: string;
    readonly title: string;
    readonly snippet: string;
    readonly label?: string | null;
    readonly authority?: Authority;
  }[];
  /** Each branch's research, whole. Absent from results stored before it was added. */
  readonly research_results?: readonly {
    readonly subtask_id: string;
    readonly findings: string;
    readonly error: string | null;
  }[];
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
