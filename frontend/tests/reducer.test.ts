/**
 * The reducer, against event sequences the backend actually produced.
 *
 * Fixtures are recorded by `backend/scripts/capture_fixtures.py` running the
 * real graph, runner and nodes. Hand-written events would test the reducer
 * against my beliefs about the backend; these test it against the backend.
 *
 * Assertions compare whole state objects, not individual fields. "The node has
 * status running" passes whether or not four other lanes are also wrong; the
 * exact map does not.
 */

import { describe, expect, it } from 'vitest';

import accepted from './fixtures/accepted.json';
import compliance from './fixtures/compliance.json';
import exhaustion from './fixtures/exhaustion.json';
import partialFailure from './fixtures/partial-failure.json';
import retry from './fixtures/retry.json';
import revision from './fixtures/revision.json';
import runFailed from './fixtures/run-failed.json';

import type { AgentEvent } from '../src/lib/events';
import {
  CRITIC,
  PLANNER,
  RESEARCHER,
  SYNTHESIZER,
  acceptAll,
  initialState,
  isScripted,
  resumeFrom,
  type BranchState,
  type NodeStatus,
  type RunState,
} from '../src/lib/reducer';

const SKELETON = [PLANNER, RESEARCHER, SYNTHESIZER, CRITIC];

// The fixtures are produced by the backend's own pydantic models, so the shape
// is guaranteed by the generator rather than by a cast at every use site.
const ACCEPTED = accepted as readonly AgentEvent[];
const PARTIAL_FAILURE = partialFailure as readonly AgentEvent[];
const RETRY = retry as readonly AgentEvent[];
const EXHAUSTION = exhaustion as readonly AgentEvent[];
const RUN_FAILED_EVENTS = runFailed as readonly AgentEvent[];
const REVISION = revision as readonly AgentEvent[];
const COMPLIANCE = compliance as readonly AgentEvent[];

function run(events: readonly AgentEvent[]): RunState {
  return acceptAll(initialState(SKELETON), events);
}

/** Node statuses as one comparable map. */
function statuses(state: RunState): Record<string, NodeStatus> {
  return Object.fromEntries(Object.values(state.nodes).map((node) => [node.id, node.status]));
}

/** Branch statuses keyed by subtask id. */
function branchStatuses(state: RunState): Record<string, NodeStatus> {
  return Object.fromEntries(state.branches.map((b) => [b.subtaskId, b.status]));
}

describe('a run the critic accepts', () => {
  it('ends with every node done and an answer', () => {
    const state = run(ACCEPTED);

    expect(statuses(state)).toEqual({
      [PLANNER]: 'done',
      [RESEARCHER]: 'idle', // the placeholder; the real work is in branches
      [SYNTHESIZER]: 'done',
      [CRITIC]: 'done',
    });
    expect(branchStatuses(state)).toEqual({
      s0: 'done',
      s1: 'done',
      s2: 'done',
      s3: 'done',
      s4: 'done',
    });
    expect(state.phase).toBe('completed');
    expect(state.iteration).toBe(1);
    expect(state.degraded).toEqual([]);
    expect(state.retryFired).toBe(false);
    expect(state.answer).toContain('Synthesised from the findings');
  });

  it('consumes every event with no gap and nothing buffered', () => {
    const state = run(ACCEPTED);

    expect(state.nextSeq).toBe(ACCEPTED.length + 1);
    expect(state.buffered).toEqual({});
    expect(state.gapAt).toBeNull();
    expect(state.degradedStream).toBe(false);
  });

  it('has exactly one researcher node before fan_out and five after', () => {
    const beforeFanOut = ACCEPTED.filter((e) => e.type !== 'fan_out' && e.seq < 4);
    const before = run(beforeFanOut);

    expect(before.fannedOut).toBe(false);
    expect(before.branches).toEqual([]);

    const after = run(ACCEPTED);
    expect(after.fannedOut).toBe(true);
    expect(after.branches.map((b) => b.subtaskId)).toEqual(['s0', 's1', 's2', 's3', 's4']);
  });

  it('labels each lane with its subtask question', () => {
    const state = run(ACCEPTED);

    expect(state.branches.map((b) => b.question)).toEqual([
      'How accurate is RAG for enterprise document QA?',
      'How accurate are long-context LLMs on the same task?',
      'What are the cost characteristics of each?',
      'What are the latency characteristics of each?',
      'When is each the better choice?',
    ]);
  });

  it('collects the sources each branch retrieved', () => {
    const state = run(ACCEPTED);

    expect(state.branches.map((b) => b.sources.length)).toEqual([2, 2, 2, 2, 2]);
    expect(state.branches[0]?.sources[0]).toEqual({
      url: 'https://example.com/how-0',
      title: 'how result 0',
      label: null,
      authority: 'secondary',
    });
  });
});

describe('a branch that fails while its siblings succeed', () => {
  // The case the whole partial-failure design exists for. Phase 1 keeps the run
  // alive by turning a dead branch into data; Phase 3 reports node_failed for it
  // anyway. If the UI renders this as a clean run, the user believes an answer
  // built on four fifths of the research is complete.
  it('shows one lane failed and four done, with the run still completing', () => {
    const state = run(PARTIAL_FAILURE);

    expect(branchStatuses(state)).toEqual({
      s0: 'done',
      s1: 'failed',
      s2: 'done',
      s3: 'done',
      s4: 'done',
    });
    expect(state.phase).toBe('completed');
    expect(state.answer).not.toBeNull();
  });

  it('carries the failure reason and reports the run as degraded', () => {
    const state = run(PARTIAL_FAILURE);
    const failed = state.branches.find((b) => b.subtaskId === 's1');

    expect(failed?.error).toContain('SearchError');
    expect(failed?.sources).toEqual([]);
    expect(state.degraded).toEqual(['researcher_failed:s1']);
  });

  it('leaves the healthy branches untouched', () => {
    const state = run(PARTIAL_FAILURE);
    const healthy = state.branches.filter((b) => b.subtaskId !== 's1');

    expect(healthy.map((b) => b.error)).toEqual([null, null, null, null]);
    expect(healthy.map((b) => b.sources.length)).toEqual([2, 2, 2, 2]);
  });
});

describe('a targeted retry', () => {
  it('re-runs only the faulted lane', () => {
    const state = run(RETRY);
    const runs: Record<string, number> = Object.fromEntries(
      state.branches.map((b) => [b.subtaskId, b.runs]),
    );

    // s2 twice, everything else once. This is targeted retry, as a number.
    expect(runs).toEqual({ s0: 1, s1: 1, s2: 2, s3: 1, s4: 1 });
  });

  it('ends with every lane done after two critic passes', () => {
    const state = run(RETRY);

    expect(branchStatuses(state)).toEqual({
      s0: 'done',
      s1: 'done',
      s2: 'done',
      s3: 'done',
      s4: 'done',
    });
    expect(state.criticRounds).toEqual([
      {
        iteration: 1,
        overall: 'revise',
        faultedSubtaskIds: ['s2'],
        perSubtask: [
          {
            subtask_id: 's2',
            verdict: 'revise',
            reason: 'the claims are not supported by the listed sources',
            guidance: 'Search for primary benchmark data rather than vendor posts.',
          },
        ],
        synthesisFeedback: null,
      },
      { iteration: 2, overall: 'accept', faultedSubtaskIds: [], perSubtask: [], synthesisFeedback: null },
    ]);
    expect(state.iteration).toBe(2);
  });

  it('latches retryFired so the edge stays legible after the fact', () => {
    const state = run(RETRY);

    expect(state.retryFired).toBe(true);
    // The final verdict faulted nothing, but the retry still happened.
    expect(state.retryTargets).toEqual([]);
  });

  it('marks the faulted lane running again mid-retry', () => {
    // Cut the stream immediately after the faulting verdict and the re-dispatch.
    const verdict = RETRY.findIndex((e) => e.type === 'critic_verdict');
    const restart = RETRY.findIndex(
      (e, i) => i > verdict && e.type === 'node_started' && e.node === RESEARCHER,
    );
    const state = run(RETRY.slice(0, restart + 1));

    expect(branchStatuses(state)).toEqual({
      s0: 'done',
      s1: 'done',
      s2: 'running', // sent back
      s3: 'done',
      s4: 'done',
    });
    expect(state.retryTargets).toEqual(['s2']);
  });
});

describe('a critic that never accepts', () => {
  it('terminates degraded rather than failing, keeping its answer', () => {
    const state = run(EXHAUSTION);

    expect(state.phase).toBe('completed');
    expect(state.degraded).toEqual(['critic_max_iterations']);
    expect(state.answer).not.toBeNull();
    expect(state.criticRounds.map((r) => r.overall)).toEqual(['revise', 'revise']);
    expect(state.iteration).toBe(2);
  });
});

describe('a run that dies', () => {
  it('ends failed, names the error, and leaves no node spinning', () => {
    const state = run(RUN_FAILED_EVENTS);

    expect(state.phase).toBe('failed');
    expect(state.error).toContain('ProviderError');
    expect(statuses(state)).toEqual({
      [PLANNER]: 'done',
      [RESEARCHER]: 'idle',
      [SYNTHESIZER]: 'failed',
      [CRITIC]: 'idle',
    });
    // A lane left mid-flight would spin forever on screen.
    expect(Object.values(statuses(state))).not.toContain('running');
    expect(state.answer).toBeNull();
  });
});

describe('sequence handling', () => {
  it('buffers a gap instead of applying past it', () => {
    // 1,2,3 then 5 — event 4 is missing.
    const withHole = [...ACCEPTED.slice(0, 3), ACCEPTED[4]].filter(
      (e): e is AgentEvent => e !== undefined,
    );
    const state = run(withHole);

    expect(state.nextSeq).toBe(4);
    expect(Object.keys(state.buffered)).toEqual(['5']);
    expect(state.gapAt).toBe(4);
  });

  it('drains the buffer once the hole fills, converging on the same state', () => {
    const ordered = ACCEPTED.slice(0, 6);
    const shuffled = [ordered[0], ordered[1], ordered[4], ordered[2], ordered[3], ordered[5]].filter(
      (e): e is AgentEvent => e !== undefined,
    );

    expect(run(shuffled)).toEqual(run(ordered));
  });

  it('ignores duplicates so a replay overlap cannot double-count a run', () => {
    const once = run(ACCEPTED);
    const twice = acceptAll(once, ACCEPTED);

    expect(twice).toEqual(once);
    // The retry counter is what a double-apply would corrupt.
    expect(twice.branches.map((b) => b.runs)).toEqual([1, 1, 1, 1, 1]);
  });

  it('reports where to resume from', () => {
    const partial = run(ACCEPTED.slice(0, 6));
    expect(resumeFrom(partial)).toBe(6);
  });
});

describe('reconnection', () => {
  // The load-bearing one. Phase 3's whole event log exists so a client that
  // drops mid-run can come back; if the reconnected UI does not converge on the
  // same state, that machinery bought nothing.
  it('converges on exactly the uninterrupted state', () => {
    const uninterrupted = run(ACCEPTED);

    // Drop after 6 events, reconnect, and replay from the last contiguous seq —
    // which overlaps by one, exactly as the server's replay does.
    const beforeDrop = run(ACCEPTED.slice(0, 6));
    const resume = resumeFrom(beforeDrop);
    const replayed = ACCEPTED.filter((e) => e.seq > resume);
    const reconnected = acceptAll(beforeDrop, replayed);

    expect(reconnected).toEqual(uninterrupted);
  });

  it('converges from a drop during the fan-out, where branches are in flight', () => {
    const uninterrupted = run(PARTIAL_FAILURE);

    // Cut at the first point where a lane is genuinely in flight, found rather
    // than guessed — a hardcoded index silently stops testing what it claims to
    // the moment the fixture changes length.
    const cut = PARTIAL_FAILURE.findIndex(
      (_event, index) =>
        Object.values(branchStatuses(run(PARTIAL_FAILURE.slice(0, index + 1)))).includes('running'),
    );
    expect(cut).toBeGreaterThan(0);

    const beforeDrop = run(PARTIAL_FAILURE.slice(0, cut + 1));
    expect(Object.values(branchStatuses(beforeDrop))).toContain('running');

    const replayed = PARTIAL_FAILURE.filter((e) => e.seq > resumeFrom(beforeDrop));
    expect(acceptAll(beforeDrop, replayed)).toEqual(uninterrupted);
  });

  it('converges when the replay overlaps generously', () => {
    // A client that resumes from further back than necessary must not end up in
    // a different state — duplicate suppression is what guarantees it.
    const uninterrupted = run(RETRY);
    const beforeDrop = run(RETRY.slice(0, 15));
    const replayed = RETRY.filter((e) => e.seq > 5); // deliberately too far back

    expect(acceptAll(beforeDrop, replayed)).toEqual(uninterrupted);
  });
});

describe('branch identity', () => {
  it('keeps every lane distinct through concurrent interleaving', () => {
    const state = run(ACCEPTED);
    const ids = state.branches.map((b: BranchState) => b.subtaskId);

    expect(new Set(ids).size).toBe(ids.length);
    expect(state.branches.every((b) => b.id === `${RESEARCHER}:${b.subtaskId}`)).toBe(true);
  });
});

describe('a revision of one branch', () => {
  // The revision's own stream: the four carried branches did their work in the
  // run being revised and send no events here. Only revision_started knows
  // they exist; without it the run would be drawn one branch wide.
  it('draws every branch, carried or edited, and re-runs only the edited one', () => {
    const state = run(REVISION);

    expect(state.branches.map((b) => [b.subtaskId, b.origin, b.status, b.runs])).toEqual([
      ['s0', 'carried', 'done', 0],
      ['s1', 'carried', 'done', 0],
      ['s2', 'edited', 'done', 1],
      ['s3', 'carried', 'done', 0],
      ['s4', 'carried', 'done', 0],
    ]);
    expect(state.phase).toBe('completed');
  });

  it('remembers what was edited, and what it was before', () => {
    const state = run(REVISION);

    expect(state.revision).toEqual({
      parentRunId: 'revision',
      subtaskId: 's2',
      before: 'What are the cost characteristics of each?',
      after: 'What are the per-token costs of long-context models?',
    });
    expect(state.branches[2]?.question).toBe('What are the per-token costs of long-context models?');
  });

  it("keeps a carried branch's sources from the run it came from", () => {
    const state = run(REVISION);

    expect(state.branches[0]?.sources.map((s) => s.url)).toEqual([
      'https://example.com/how-0',
      'https://example.com/how-1',
    ]);
  });
});

describe('a compliance run', () => {
  it('checks every draft and keeps each result in order', () => {
    const state = run(COMPLIANCE);

    expect(state.citationChecks.map((c) => [c.ok, c.unresolved])).toEqual([
      [false, ['AI Act Art. 114']],
      [true, []],
    ]);
  });

  it('keeps the law apart from the web in each branch', () => {
    const state = run(COMPLIANCE);
    const first = state.branches[0]?.sources[0];

    expect(first?.label).toBe('AI Act Annex III(4)(a)');
    expect(first?.authority).toBe('primary_law');
  });
});

describe('a scripted run', () => {
  it('is told apart by its model calls', () => {
    // The recorded runs were made with scripted models, and their calls say so.
    expect(isScripted(run(COMPLIANCE))).toBe(true);
    expect(isScripted({ ...run(COMPLIANCE), calls: run(COMPLIANCE).calls.filter((c) => c.kind === 'tool') })).toBe(false);
  });
});

describe('the support check', () => {
  it('keeps each round reading the citations that passed the law check', () => {
    const state = run(COMPLIANCE);

    // The first draft cited two real provisions for what they do not say; the
    // rewrite kept the one that does. The invented article is never read.
    expect(
      state.supportChecks.map((s) => [s.iteration, s.supported.map((e) => e.label), s.unsupported.map((e) => e.label)]),
    ).toEqual([
      [1, ['AI Act Annex III(4)(a)'], ['AI Act Annex VIII(7)', 'AI Act Art. 16(b)']],
      [2, ['AI Act Annex III(4)(a)'], []],
    ]);
  });
});

describe('what the inspector reads', () => {
  it('records every agent call with the node and span that made it', () => {
    const state = run(ACCEPTED);
    const researcher = state.calls.filter((c) => c.node === RESEARCHER);

    // Each branch searched once and asked the model once, both in its one span.
    expect(researcher.map((c) => [c.subtask_id, c.kind, c.span_id]).sort()).toEqual(
      state.branches
        .flatMap((b) => [
          [b.subtaskId, 'model', b.spans[0]],
          [b.subtaskId, 'tool', b.spans[0]],
        ])
        .sort(),
    );
    expect(state.calls.filter((c) => c.node !== RESEARCHER).map((c) => [c.node, c.kind, typeof c.span_id])).toEqual([
      [PLANNER, 'model', 'string'],
      [SYNTHESIZER, 'model', 'string'],
      [CRITIC, 'model', 'string'],
    ]);
  });

  it('gives a retried branch one span per attempt', () => {
    const state = run(RETRY);

    expect(state.branches.map((b) => b.spans.length)).toEqual([1, 1, 2, 1, 1]);
  });

  it('keeps every applied event on the timeline, in order', () => {
    const state = run(ACCEPTED);

    expect(state.timeline.map((e) => e.seq)).toEqual(ACCEPTED.map((e) => e.seq));
  });

  it('builds the same timeline from a replay that overlaps what it already has', () => {
    const live = run(ACCEPTED);
    const replayed = acceptAll(run(ACCEPTED.slice(0, 10)), ACCEPTED.slice(5));

    expect(replayed.timeline).toEqual(live.timeline);
  });
});

describe('an event type this client does not know', () => {
  it('is ignored, not fatal: the next event still applies', () => {
    const unknown = { ...ACCEPTED[1], type: 'from_a_newer_backend', payload: {} } as unknown as AgentEvent;
    const events = [ACCEPTED[0], unknown, ...ACCEPTED.slice(1).map((e) => ({ ...e, seq: e.seq + 1 }))];

    const state = run(events as readonly AgentEvent[]);

    expect(state.phase).toBe('completed');
    expect(state.branches.length).toBe(5);
  });
});

