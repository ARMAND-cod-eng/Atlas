// @vitest-environment jsdom
/**
 * The workspace's keys: j and k step through the agents in the order they
 * ran, e opens the selected branch's question for editing, Escape closes the
 * inspector. Never while someone is typing.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import retry from '../fixtures/retry.json';

import type { AgentEvent } from '../../src/lib/events';
import {
  CRITIC,
  PLANNER,
  RESEARCHER,
  SYNTHESIZER,
  acceptAll,
  initialState,
} from '../../src/lib/reducer';
import { agentOrder, stepAgent, useShortcuts } from '../../src/lib/shortcuts';

const state = acceptAll(
  initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]),
  retry as readonly AgentEvent[],
);

afterEach(cleanup);

describe('the order j and k step through', () => {
  it('is the agents in the order they first ran', () => {
    expect(agentOrder(state)).toEqual([
      PLANNER,
      'researcher:s0',
      'researcher:s1',
      'researcher:s2',
      'researcher:s3',
      'researcher:s4',
      SYNTHESIZER,
      CRITIC,
    ]);
  });

  it('starts at the first agent, and stops at either end', () => {
    const order = agentOrder(state);

    expect(stepAgent(order, null, 1)).toBe(PLANNER);
    expect(stepAgent(order, PLANNER, 1)).toBe('researcher:s0');
    expect(stepAgent(order, PLANNER, -1)).toBe(PLANNER);
    expect(stepAgent(order, CRITIC, 1)).toBe(CRITIC);
    expect(stepAgent(order, null, -1)).toBe(CRITIC);
  });
});

function Harness({ onEdit }: { readonly onEdit: () => void }): React.JSX.Element {
  const [selected, setSelected] = useState<string | null>(null);
  useShortcuts({ order: agentOrder(state), selected, onSelect: setSelected, onEdit });
  return (
    <div>
      <p data-testid="selected">{selected ?? 'none'}</p>
      <textarea aria-label="typing here" />
    </div>
  );
}

describe('the keys', () => {
  it('step, edit and close', () => {
    const onEdit = vi.fn();
    render(<Harness onEdit={onEdit} />);
    const shown = () => screen.getByTestId('selected').textContent;

    fireEvent.keyDown(window, { key: 'j' });
    fireEvent.keyDown(window, { key: 'j' });
    expect(shown()).toBe('researcher:s0');
    fireEvent.keyDown(window, { key: 'k' });
    expect(shown()).toBe(PLANNER);
    fireEvent.keyDown(window, { key: 'e' });
    expect(onEdit).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(shown()).toBe('none');
  });

  it('are left alone while someone types, or with a modifier held', () => {
    render(<Harness onEdit={() => undefined} />);

    fireEvent.keyDown(screen.getByLabelText('typing here'), { key: 'j' });
    fireEvent.keyDown(window, { key: 'j', ctrlKey: true });
    fireEvent.keyDown(window, { key: 'j', metaKey: true });

    expect(screen.getByTestId('selected').textContent).toBe('none');
  });
});
