'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import {
  CRITIC_VERDICT,
  FAN_OUT,
  ITERATION_STARTED,
  NODE_COMPLETED,
  NODE_FAILED,
  NODE_STARTED,
  RUN_COMPLETED,
  RUN_FAILED,
  RUN_STARTED,
  type AgentEvent,
  type EventType,
  type RunResult,
  type Topology,
  isTerminal,
} from './events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, accept, initialState, isFinished, resumeFrom, type RunState } from './reducer';

/**
 * 127.0.0.1, not localhost — and this is worth two seconds per request.
 *
 * On Windows `localhost` resolves to `::1` before `127.0.0.1`, and uvicorn binds
 * IPv4 only. The browser therefore tries IPv6 first, waits for that attempt to
 * fail, and only then connects — measured at **2,050 ms against 22 ms** for the
 * identical request to `127.0.0.1`. Every call pays it: the topology fetch, the
 * POST, and the SSE connection, which is why the run page appeared to take many
 * seconds to show anything.
 *
 * Naming the loopback address explicitly removes the resolution step entirely.
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

const SKELETON = [PLANNER, RESEARCHER, SYNTHESIZER, CRITIC];

/**
 * Every named event we subscribe to.
 *
 * `EventSource.onmessage` only fires for frames with no `event:` field, and ours
 * all have one — so each type needs an explicit listener. Listed rather than
 * inferred so a new backend event type shows up as a missing listener during
 * review, not as an event that silently never arrives.
 */
const EVENT_TYPES: readonly EventType[] = [
  RUN_STARTED,
  NODE_STARTED,
  NODE_COMPLETED,
  NODE_FAILED,
  FAN_OUT,
  ITERATION_STARTED,
  CRITIC_VERDICT,
  RUN_COMPLETED,
  RUN_FAILED,
];

export type Connection = 'connecting' | 'open' | 'reconnecting' | 'closed';

export interface RunStream {
  readonly state: RunState;
  readonly connection: Connection;
  readonly result: RunResult | null;
}

export function useTopology(): Topology | null {
  const [topology, setTopology] = useState<Topology | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetch(`${API_BASE}/graph/topology`)
      .then((response) => (response.ok ? (response.json() as Promise<Topology>) : null))
      .then((value) => {
        if (!cancelled && value !== null) setTopology(value);
      })
      .catch(() => {
        // The graph simply does not render without it; the page says so.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return topology;
}

/**
 * Subscribe to a run's events and fold them into state.
 *
 * Three things here are not obvious:
 *
 * **The stream is closed explicitly on a terminal event.** When a server ends an
 * SSE response normally, `EventSource` treats it as a dropped connection and
 * reconnects — so without this, every finished run leaves the browser opening an
 * endless series of idle streams that sit on heartbeats forever.
 *
 * **A gap gets exactly one recovery attempt.** `eventlog.append` fails open on
 * the backend, so a Redis hiccup means an event is permanently gone. Retrying
 * forever would spin on a hole that will never fill, so the second sighting of
 * the same missing `seq` stops the loop and marks the stream degraded instead.
 *
 * **Reconnection is the browser's job, not ours.** `EventSource` retries on its
 * own and resends `Last-Event-ID`; we only intervene for a gap, where we need
 * the query-parameter form because `EventSource` cannot set headers.
 */
export function useRunStream(runId: string): RunStream {
  const [state, setState] = useState<RunState>(() => initialState(SKELETON));
  const [connection, setConnection] = useState<Connection>('connecting');
  const [result, setResult] = useState<RunResult | null>(null);

  // Refs, not state: these drive the connection lifecycle and must not
  // re-trigger the effect that owns it.
  const sourceRef = useRef<EventSource | null>(null);
  const recoveredAtRef = useRef<number | null>(null);
  const resumeRef = useRef<number>(0);

  const close = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
  }, []);

  useEffect(() => {
    let disposed = false;

    const connect = (afterSeq: number): void => {
      if (disposed) return;
      const url = new URL(`${API_BASE}/runs/${runId}/events`);
      if (afterSeq > 0) url.searchParams.set('last_event_id', String(afterSeq));

      const source = new EventSource(url.toString());
      sourceRef.current = source;

      source.onopen = () => {
        if (!disposed) setConnection('open');
      };

      source.onerror = () => {
        if (disposed) return;
        // readyState CONNECTING means the browser is retrying by itself, which
        // is the normal path for a transient drop. CLOSED means it gave up.
        setConnection(source.readyState === EventSource.CLOSED ? 'closed' : 'reconnecting');
      };

      const handle = (raw: MessageEvent<string>): void => {
        if (disposed) return;
        const event = JSON.parse(raw.data) as AgentEvent;

        setState((current) => {
          const next = accept(current, event);
          resumeRef.current = resumeFrom(next);

          if (next.gapAt !== null) {
            if (recoveredAtRef.current === next.gapAt) {
              // Same hole, second time. It is not coming.
              return { ...next, degradedStream: true };
            }
            recoveredAtRef.current = next.gapAt;
            // Reopen from the last contiguous seq, outside the state update.
            queueMicrotask(() => {
              if (disposed) return;
              close();
              connect(next.nextSeq - 1);
            });
          }
          return next;
        });

        if (isTerminal(event.type)) {
          close();
          setConnection('closed');
        }
      };

      for (const type of EVENT_TYPES) {
        source.addEventListener(type, handle as EventListener);
      }
    };

    connect(0);

    return () => {
      disposed = true;
      close();
    };
  }, [runId, close]);

  // The terminal event carries a summary; the full sources list and per-call
  // metrics are fetched once from here rather than replayed on every reconnect.
  useEffect(() => {
    if (!isFinished(state)) return;
    let cancelled = false;
    void fetch(`${API_BASE}/runs/${runId}`)
      .then((response) => (response.ok ? (response.json() as Promise<{ result: RunResult | null }>) : null))
      .then((body) => {
        if (!cancelled && body?.result) setResult(body.result);
      })
      .catch(() => {
        // The answer is already on screen from run_completed; this only adds
        // the full source list and metrics.
      });
    return () => {
      cancelled = true;
    };
  }, [runId, state]);

  return { state, connection, result };
}

/** A ticking clock for elapsed times. Stops when nothing is running. */
export function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [active]);

  return now;
}
