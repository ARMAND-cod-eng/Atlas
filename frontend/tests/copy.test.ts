/**
 * The answer as Markdown, to paste elsewhere: the question, the answer as
 * written, each provision it cites with its link, and the boundary.
 */

import { describe, expect, it } from 'vitest';

import type { SourceRef } from '../src/lib/events';
import { answerMarkdown } from '../src/lib/copy';

const ART_6 = 'https://eur-lex.europa.eu/eli/reg/2024/1689/oj#art_6';
const ANNEX = 'https://eur-lex.europa.eu/eli/reg/2024/1689/oj#anx_III';

const SOURCES: readonly SourceRef[] = [
  { url: ART_6, title: 'Classification rules for high-risk AI systems', label: 'AI Act Art. 6(2)' },
  { url: ANNEX, title: 'High-risk AI systems referred to in Article 6(2)', label: 'AI Act Annex III(4)(a)' },
  { url: 'https://example.com/guide', title: 'A guide', label: null },
];

describe('the answer as Markdown', () => {
  it('links each provision it cites, once, in the order cited, and keeps the boundary', () => {
    const markdown = answerMarkdown({
      question: 'Is a CV-screening tool high-risk?',
      answer: 'Yes [AI Act Annex III(4)(a)], as Art. 6(2) says [AI Act Art. 6(2)]. Again [AI Act Art. 6(2)].',
      sources: SOURCES,
      lawAsOf: '23 October 2026',
    });

    expect(markdown).toBe(
      [
        '## Is a CV-screening tool high-risk?',
        '',
        'Yes [AI Act Annex III(4)(a)], as Art. 6(2) says [AI Act Art. 6(2)]. Again [AI Act Art. 6(2)].',
        '',
        'Provisions cited:',
        '',
        '- [AI Act Annex III(4)(a)] — High-risk AI systems referred to in Article 6(2)',
        '- [AI Act Art. 6(2)] — Classification rules for high-risk AI systems',
        '',
        `[AI Act Annex III(4)(a)]: <${ANNEX}>`,
        `[AI Act Art. 6(2)]: <${ART_6}>`,
        '',
        '_A research aid, not legal advice: the law as published in the Official Journal, retrieved 23 October 2026._',
        '',
      ].join('\n'),
    );
  });

  it('lists a provision it could not link as such, rather than dropping it', () => {
    const markdown = answerMarkdown({
      question: null,
      answer: 'Fines apply [AI Act Art. 99(4); GDPR Art. 83(5)].',
      sources: SOURCES,
      lawAsOf: '23 October 2026',
    });

    expect(markdown.split('\n').slice(0, 7)).toEqual([
      'Fines apply [AI Act Art. 99(4); GDPR Art. 83(5)].',
      '',
      'Provisions cited:',
      '',
      '- AI Act Art. 99(4) — not among the sources this run retrieved',
      '- GDPR Art. 83(5) — not among the sources this run retrieved',
      '',
    ]);
    // No link is made to a source the run never had.
    expect(markdown).not.toContain(']: <');
  });

  it('names a provision with no title of its own once, a recital', () => {
    const recital = 'http://data.europa.eu/eli/reg/2016/679/oj#rct_35';
    const markdown = answerMarkdown({
      question: null,
      answer: 'Health data [GDPR Recital 35].',
      sources: [{ url: recital, title: 'GDPR Recital 35', label: 'GDPR Recital 35' }],
      lawAsOf: '23 October 2026',
    });

    expect(markdown.split('\n').slice(2, 7)).toEqual([
      'Provisions cited:',
      '',
      '- [GDPR Recital 35]',
      '',
      `[GDPR Recital 35]: <${recital}>`,
    ]);
  });

  it('says nothing of provisions when the answer cites none', () => {
    const markdown = answerMarkdown({ question: null, answer: 'No.', sources: SOURCES, lawAsOf: '23 October 2026' });

    expect(markdown).toBe(
      'No.\n\n_A research aid, not legal advice: the law as published in the Official Journal, retrieved 23 October 2026._\n',
    );
  });
});
