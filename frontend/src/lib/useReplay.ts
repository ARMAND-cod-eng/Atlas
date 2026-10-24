'use client';

import { useCallback, useEffect, useState } from 'react';

import type { AgentEvent } from './events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, accept, initialState, type RunState } from './reducer';

const SKELETON = [PLANNER, RESEARCHER, SYNTHESIZER, CRITIC];

// A recorded run's real pauses, compressed: long enough to watch the fan-out
// light up, short enough that a demo does not sit through the synthesizer.
const MIN_GAP_MS = 120;
const MAX_GAP_MS = 700;

/**
 * Play a recorded event stream through the same reducer a live run uses.
 *
 * What the /demo page shows: the workspace, driven by events the backend
 * actually produced, with no backend running. Each event is applied after its
 * recorded gap from the previous one, clamped.
 */
export function useReplay(events: readonly AgentEvent[]): {
  readonly state: RunState;
  readonly done: boolean;
  readonly restart: () => void;
} {
  const [state, setState] = useState<RunState>(() => initialState(SKELETON));
  const [played, setPlayed] = useState(0);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    if (played >= events.length) return;
    const event = events[played];
    const previous = played > 0 ? events[played - 1] : undefined;
    if (event === undefined) return;
    const gap = previous === undefined ? 0 : Date.parse(event.ts) - Date.parse(previous.ts);
    const delay = Math.min(MAX_GAP_MS, Math.max(MIN_GAP_MS, gap));
    const timer = window.setTimeout(() => {
      setState((current) => accept(current, event));
      setPlayed((count) => count + 1);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [events, played, epoch]);

  const restart = useCallback(() => {
    setState(initialState(SKELETON));
    setPlayed(0);
    setEpoch((value) => value + 1);
  }, []);

  return { state, done: played >= events.length, restart };
}
