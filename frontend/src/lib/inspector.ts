/**
 * One agent, as the inspector shows it: its task, its attempts, what each
 * attempt called, and what it produced.
 *
 * Pure functions of the run state (and the finished run's result, when there is
 * one), tested against recorded streams. The prompts themselves are not here:
 * they are fetched one call at a time, when someone opens them.
 */

import type { RunResult, SourceRef } from './events';
import {
  CRITIC,
  HUMAN_EDIT,
  PLANNER,
  RESEARCHER,
  SYNTHESIZER,
  type BranchState,
  type CallEntry,
  type CriticRound,
  type RunState,
} from './reducer';

export interface Attempt {
  /** 1-based, in the order the attempts started. */
  readonly number: number;
  readonly spanId: string;
  readonly calls: readonly CallEntry[];
}

export type Agent =
  | {
      readonly kind: 'branch';
      readonly id: string;
      readonly title: string;
      readonly branch: BranchState;
      /** The question before a person rewrote it, when they did. */
      readonly before: string | null;
      readonly findings: string | null;
      readonly attempts: readonly Attempt[];
    }
  | {
      readonly kind: 'stage';
      readonly id: string;
      readonly title: string;
      readonly role: string;
      readonly attempts: readonly Attempt[];
    };

const ROLES: Readonly<Record<string, { readonly title: string; readonly role: string }>> = {
  [PLANNER]: {
    title: 'Planner',
    role: 'Splits the question into independent branches that can be researched in parallel.',
  },
  [SYNTHESIZER]: {
    title: 'Synthesizer',
    role: 'Writes one answer from every branch’s findings, citing each provision it relies on.',
  },
  [CRITIC]: {
    title: 'Critic',
    role:
      'Checks the answer against the sources that were actually retrieved, and sends back the branches or the write-up it cannot vouch for. In the compliance domain the answer’s citations are also checked against the law.',
  },
  [HUMAN_EDIT]: {
    title: 'Your edit',
    role: 'Where a person rewrites one branch’s question: that branch is researched again and the answer rewritten; the others are carried over.',
  },
};

/** A node's calls, grouped into attempts by span, in the order they started. */
export function attemptsOf(calls: readonly CallEntry[], spans: readonly string[] | null): readonly Attempt[] {
  const order = spans ?? [...new Set(calls.map((call) => call.span_id).filter((id): id is string => id !== null))];
  return order.map((spanId, index) => ({
    number: index + 1,
    spanId,
    calls: calls.filter((call) => call.span_id === spanId),
  }));
}

/** The agent a graph node id names, or null for one the run has not reached. */
export function agentOf(state: RunState, result: RunResult | null, nodeId: string): Agent | null {
  if (nodeId.startsWith(`${RESEARCHER}:`)) {
    const branch = state.branches.find((candidate) => candidate.id === nodeId);
    if (branch === undefined) return null;
    const calls = state.calls.filter(
      (call) => call.node === RESEARCHER && call.subtask_id === branch.subtaskId,
    );
    const research = result?.research_results?.find((r) => r.subtask_id === branch.subtaskId);
    const edited = state.revision?.subtaskId === branch.subtaskId ? state.revision : null;
    return {
      kind: 'branch',
      id: nodeId,
      // Numbered as the graph numbers its lanes: a planner's fallback ids are uuids.
      title: `Researcher ${state.branches.indexOf(branch) + 1}`,
      branch,
      before: edited?.before ?? null,
      findings: research?.findings ?? branch.findingsPreview,
      attempts: attemptsOf(calls, branch.spans),
    };
  }
  const known = ROLES[nodeId];
  if (known === undefined) return null;
  const calls = state.calls.filter((call) => call.node === nodeId);
  return { kind: 'stage', id: nodeId, ...known, attempts: attemptsOf(calls, null) };
}

/** A branch's sources, the law first: the order a reader should weigh them in. */
export function byAuthority(sources: readonly SourceRef[]): {
  readonly law: readonly SourceRef[];
  readonly guidance: readonly SourceRef[];
  readonly other: readonly SourceRef[];
} {
  return {
    law: sources.filter((source) => source.authority === 'primary_law'),
    guidance: sources.filter((source) => source.authority === 'official_guidance'),
    other: sources.filter(
      (source) => source.authority !== 'primary_law' && source.authority !== 'official_guidance',
    ),
  };
}

/** What the critic said in each round, about one branch or about the write-up. */
export function reviewsOf(rounds: readonly CriticRound[], subtaskId: string | null): readonly {
  readonly iteration: number;
  readonly verdict: 'accept' | 'revise';
  readonly reason: string | null;
  readonly guidance: string | null;
}[] {
  return rounds.map((round) => {
    const mine = subtaskId === null ? undefined : round.perSubtask.find((v) => v.subtask_id === subtaskId);
    if (subtaskId === null) {
      return {
        iteration: round.iteration,
        verdict: round.overall,
        reason: round.synthesisFeedback,
        guidance: null,
      };
    }
    return {
      iteration: round.iteration,
      verdict: mine?.verdict ?? 'accept',
      reason: mine?.reason ?? null,
      guidance: mine?.guidance ?? null,
    };
  });
}
