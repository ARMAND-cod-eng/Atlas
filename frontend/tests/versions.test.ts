/**
 * A thread's runs as versions of its answer, and what changed between two.
 */

import { describe, expect, it } from 'vitest';

import type { RunVersion } from '../src/lib/api';
import { answerDiff, comparable, versionRows } from '../src/lib/versions';

function run(overrides: Partial<RunVersion> & Pick<RunVersion, 'run_id'>): RunVersion {
  return {
    kind: 'question',
    query: 'Is a CV-screening tool high-risk?',
    parent_run_id: null,
    revision: null,
    status: 'completed',
    created_at: '2026-10-24T10:00:00Z',
    finished_at: '2026-10-24T10:01:00Z',
    ...overrides,
  };
}

const THREAD: readonly RunVersion[] = [
  run({ run_id: 'a' }),
  run({
    run_id: 'b',
    kind: 'revision',
    parent_run_id: 'a',
    revision: { parent_run_id: 'a', subtask_id: 's2', before: 'What must a provider do?', after: 'What must a deployer do?' },
  }),
  run({ run_id: 'c', query: 'And for a deployer?', parent_run_id: 'b', status: 'failed' }),
  run({ run_id: 'd', query: 'What about GPAI?' }),
];

describe('versions', () => {
  it('number the runs in order and say what each was', () => {
    expect(
      versionRows(THREAD, 'b').map((row) => [row.number, row.title, row.detail, row.before, row.basedOn, row.current]),
    ).toEqual([
      [1, 'Question', 'Is a CV-screening tool high-risk?', null, null, false],
      [2, 'Edited s2', 'What must a deployer do?', 'What must a provider do?', 1, true],
      [3, 'Follow-up', 'And for a deployer?', null, 2, false],
      [4, 'Follow-up', 'What about GPAI?', null, null, false],
    ]);
  });

  it('offer the finished others for comparison, the one revised first', () => {
    const rows = versionRows([THREAD[3], THREAD[0], THREAD[1], THREAD[2]].filter((r) => r !== undefined), 'b');

    expect(comparable(rows).map((row) => row.runId)).toEqual(['a', 'd']);
  });
});

describe('a diff of two answers', () => {
  it('marks the words removed and added, and counts them', () => {
    expect(answerDiff('The provider must register it.', 'The deployer must monitor and register it.')).toEqual({
      parts: [
        { kind: 'same', text: 'The ' },
        { kind: 'removed', text: 'provider' },
        { kind: 'added', text: 'deployer' },
        { kind: 'same', text: ' must ' },
        { kind: 'added', text: 'monitor and ' },
        { kind: 'same', text: 'register it.' },
      ],
      added: 3,
      removed: 1,
    });
  });

  it('is all the same for one answer twice', () => {
    expect(answerDiff('Yes [AI Act Art. 6(2)].', 'Yes [AI Act Art. 6(2)].')).toEqual({
      parts: [{ kind: 'same', text: 'Yes [AI Act Art. 6(2)].' }],
      added: 0,
      removed: 0,
    });
  });
});
