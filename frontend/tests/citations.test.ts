/**
 * Citations in an answer: the port of the backend's parser, held to the
 * backend's own answers, and the marking that turns them into chips.
 */

import { describe, expect, it } from 'vitest';

import cases from './fixtures/citation-cases.json';
import compliance from './fixtures/compliance.json';

import { CITE_PREFIX, citedLabels, judgementsOf, markCitations, statusOf, supportFor } from '../src/lib/citations';
import type { AgentEvent } from '../src/lib/events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, acceptAll, initialState } from '../src/lib/reducer';

describe('the parser', () => {
  it.each(cases)('reads $text as the backend does', ({ text, labels }) => {
    expect(citedLabels(text)).toEqual(labels);
  });
});

describe('marking an answer', () => {
  it('makes each citation a link to its label', () => {
    const marked = markCitations('High-risk [AI Act Art. 6(2); AI Act Annex III(4)(a)], see [1].');

    expect(marked).toEqual({
      markdown: `High-risk [AI Act Art. 6(2)](${CITE_PREFIX}0) [AI Act Annex III(4)(a)](${CITE_PREFIX}1), see [1].`,
      labels: ['AI Act Art. 6(2)', 'AI Act Annex III(4)(a)'],
    });
  });

  it('reads two brackets side by side as two citations', () => {
    expect(markCitations('[AI Act Art. 6(2)][AI Act Art. 9]').labels).toEqual(['AI Act Art. 6(2)', 'AI Act Art. 9']);
  });

  it("leaves Markdown's own links alone", () => {
    const text = 'Read [GDPR Art. 35](https://eur-lex.europa.eu) and ![GDPR Art. 5](x.png).';

    expect(markCitations(text)).toEqual({ markdown: text, labels: [] });
  });
});

describe("a citation's status", () => {
  const check = {
    ok: false,
    cited: ['AI Act Art. 6(2)', 'AI Act Art. 99(9)', 'GDPR Art. 35(1)', 'GDPR Art. 5'],
    unresolved: ['AI Act Art. 99(9)'],
    unretrieved: ['GDPR Art. 35(1)'],
    misquoted: [['GDPR Art. 5', 'lawfully and fairly']] as const,
    ungroundedUrls: [],
  };

  it('is what the last check said', () => {
    expect(
      ['AI Act Art. 6(2)', 'AI Act Art. 99(9)', 'GDPR Art. 35(1)', 'GDPR Art. 5', 'GDPR Art. 6'].map((label) =>
        statusOf(label, check),
      ),
    ).toEqual(['verified', 'unknown', 'unretrieved', 'misquoted', 'unchecked']);
  });

  it('is unchecked where no check ran', () => {
    expect(statusOf('AI Act Art. 6(2)', null)).toBe('unchecked');
  });

  it("matches the recorded run's check, label for label", () => {
    const state = acceptAll(
      initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]),
      compliance as readonly AgentEvent[],
    );
    const last = state.citationChecks.at(-1) ?? null;
    const { labels } = markCitations(state.answer ?? '');

    expect(labels.length).toBeGreaterThan(0);
    // The check lists each provision once; the answer may cite one twice.
    expect([...new Set(labels)]).toEqual(last?.cited);
  });
});

describe('the support check', () => {
  const check = {
    ok: true,
    cited: ['AI Act Art. 2(3)', 'AI Act Art. 2(1)(c)', 'GDPR Art. 5'],
    unresolved: [],
    unretrieved: [],
    misquoted: [],
    ungroundedUrls: [],
  };
  const support = {
    iteration: 1,
    supported: [{ label: 'AI Act Art. 2(1)(c)', sentence: 'Outside the EU too.', quote: 'located in a third country', reason: '' }],
    unsupported: [{ label: 'AI Act Art. 2(3)', sentence: 'Outside the EU too.', quote: '', reason: 'national security' }],
    unverified: [],
    unchecked: 0,
  };

  it('marks what the reading found, and leaves a citation it did not read as verified', () => {
    expect(
      ['AI Act Art. 2(3)', 'AI Act Art. 2(1)(c)', 'GDPR Art. 5'].map((label) => statusOf(label, check, support)),
    ).toEqual(['unsupported', 'supported', 'verified']);
    expect(judgementsOf('AI Act Art. 2(3)', support).map((e) => e.reason)).toEqual(['national security']);
  });

  it('never lets the reading overrule the law check', () => {
    const invented = { ...check, unresolved: ['AI Act Art. 2(3)'] };

    expect(statusOf('AI Act Art. 2(3)', invented, support)).toBe('unknown');
  });

  it("uses only the reading of the draft on screen", () => {
    const twoDrafts = [check, check];

    expect(supportFor([check], [support])).toEqual(support);
    // Draft 2 is on screen; round 1 read draft 1, and round 2 has not happened.
    expect(supportFor(twoDrafts, [support])).toBeNull();
    expect(supportFor(twoDrafts, [support, { ...support, iteration: 2 }])?.iteration).toBe(2);
  });
});
