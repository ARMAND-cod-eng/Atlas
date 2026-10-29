/**
 * A run's status as one short phrase, for the header's pill: what it is doing,
 * and for how long. A finished run's time comes from its own timestamps, so a
 * replay reads as the run did, not as the wall clock does.
 */

import type { Tone } from '@/components/ui';

import { durationSeconds, elapsedSeconds, formatSeconds } from './format';
import type { RunState } from './reducer';

function ticking(seconds: number): string {
  const whole = Math.floor(seconds);
  return whole < 60 ? `${whole}s` : `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, '0')}s`;
}

export function runStatus(state: RunState, now: number): { readonly tone: Tone; readonly text: string } {
  if (state.phase === 'failed') return { tone: 'failed', text: 'Failed' };
  if (state.phase === 'completed') {
    return { tone: 'done', text: `Complete · ${formatSeconds(durationSeconds(state.startedAt, state.finishedAt))}` };
  }
  const elapsed = elapsedSeconds(state.startedAt, now);
  if (state.phase === 'connecting' || elapsed === null) return { tone: 'neutral', text: 'Starting' };
  return { tone: 'running', text: `Running · ${ticking(elapsed)}` };
}
