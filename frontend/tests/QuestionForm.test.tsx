// @vitest-environment jsdom
/**
 * The question box's examples, and the date the disclaimer gives for the law.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import manifest from '../../backend/atlas/library/sources/manifest.json';
import { LAW_AS_OF } from '../src/components/Disclaimer';
import { QuestionForm } from '../src/components/QuestionForm';

afterEach(cleanup);

describe('an example question', () => {
  it('goes in the box on a click, ready to edit or send', () => {
    render(
      <QuestionForm
        placeholder="Ask"
        examples={['Is a CV-screening tool high-risk?']}
        hint=""
        submitLabel="Research"
        onStarted={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Is a CV-screening tool high-risk?' }));

    expect(screen.getByLabelText('Your question')).toHaveProperty('value', 'Is a CV-screening tool high-risk?');
    expect(screen.getByRole('button', { name: 'Research' })).toHaveProperty('disabled', false);
  });
});

describe('the disclaimer', () => {
  it('dates the law as the library retrieved it', () => {
    const dates = new Set(Object.values(manifest).map((act) => act.retrieved_at));
    const formatted = [...dates].map((date) =>
      new Date(`${date}T00:00:00Z`).toLocaleDateString('en-GB', {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
        timeZone: 'UTC',
      }),
    );

    expect(formatted).toEqual([LAW_AS_OF]);
  });
});
