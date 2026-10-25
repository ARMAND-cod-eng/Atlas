/**
 * The open agent and tab as a URL: read back what was written, and leave the
 * rest of the URL alone.
 */

import { describe, expect, it } from 'vitest';

import { readLink, writeLink } from '../src/lib/inspectorLink';

const PAGE = 'http://127.0.0.1:3000/runs/r1?from=r0';

describe('an inspector link', () => {
  it('opens an agent on a tab', () => {
    const href = writeLink(PAGE, { node: 'researcher:s1', tab: 'prompt' });

    expect(href).toBe('http://127.0.0.1:3000/runs/r1?from=r0&node=researcher%3As1&tab=prompt');
    expect(readLink(new URL(href).search)).toEqual({ node: 'researcher:s1', tab: 'prompt' });
  });

  it('leaves the default tab out', () => {
    expect(writeLink(PAGE, { node: 'planner', tab: 'task' })).toBe(`${PAGE}&node=planner`);
  });

  it('closing the inspector removes both, and keeps the rest', () => {
    const open = writeLink(PAGE, { node: 'critic', tab: 'output' });

    expect(writeLink(open, { node: null, tab: 'output' })).toBe(PAGE);
  });

  it('reads an unknown tab as the first one', () => {
    expect(readLink('?node=critic&tab=secrets')).toEqual({ node: 'critic', tab: 'task' });
  });
});
