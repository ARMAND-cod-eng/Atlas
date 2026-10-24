import { describe, expect, it } from 'vitest';

import { recordMeasurements, type Measurements } from '../src/lib/measure';

const EMPTY: Measurements = new Map();

describe('recording node sizes', () => {
  it('keeps the size React Flow reports for each node', () => {
    const after = recordMeasurements(EMPTY, [
      { type: 'dimensions', id: 'planner', dimensions: { width: 210, height: 72 } },
      { type: 'dimensions', id: 'critic', dimensions: { width: 210, height: 88 } },
    ]);

    expect([...after]).toEqual([
      ['planner', { width: 210, height: 72 }],
      ['critic', { width: 210, height: 88 }],
    ]);
  });

  it('returns the same map when nothing changed, so a report cannot loop renders', () => {
    const known = recordMeasurements(EMPTY, [
      { type: 'dimensions', id: 'planner', dimensions: { width: 210, height: 72 } },
    ]);

    const again = recordMeasurements(known, [
      { type: 'dimensions', id: 'planner', dimensions: { width: 210, height: 72 } },
      { type: 'select', id: 'planner' },
    ]);

    expect(again).toBe(known);
  });

  it('takes a new size when a node grows', () => {
    const known = recordMeasurements(EMPTY, [
      { type: 'dimensions', id: 'researcher:s0', dimensions: { width: 210, height: 72 } },
    ]);

    const grown = recordMeasurements(known, [
      { type: 'dimensions', id: 'researcher:s0', dimensions: { width: 210, height: 96 } },
    ]);

    expect(grown.get('researcher:s0')).toEqual({ width: 210, height: 96 });
    expect(known.get('researcher:s0')).toEqual({ width: 210, height: 72 });
  });
});
