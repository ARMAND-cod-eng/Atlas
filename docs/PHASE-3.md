# Atlas — Phase 3: SSE streaming and live agent-graph events

## Context

Phase 2 closed the core architecture but left all of it invisible. A client posted a query, waited
67 seconds at a blank screen, and got an answer. There was no way to see the plan, watch five
branches run, or know the critic had rejected something and sent a branch back.

Worse, the run belonged to the HTTP request. A client that dropped at second 40 of a 67-second run
destroyed the run and the tokens already spent, and reconnecting started from zero.

This phase makes the run observable and makes it outlive the connection. Backend only.

---

## Design decision: where the run executes

**Shipped: (b) — an arq worker executes the run, events go to a Redis Stream, and a separate SSE
endpoint relays them.**

The obvious framing is (a) in-request vs (b) decoupled, but there is a middle option that changes
how much (b) actually costs, and it deserves naming.

**(a) strictly in-request** — rejected. A disconnect at 40s cancels the run.

**(a′) in-process background task** — the graph runs as an `asyncio.Task` owned by the app rather
than by the request; events still go to a Redis Stream; the SSE endpoint is still a separate relay.
This is "(a) plus reconnection handling" done properly, and it gets ~90% of the value: the run
survives disconnect, reconnection replays exactly, a backgrounded mobile tab is harmless, and any
API instance can serve the stream because the events live in Redis rather than in a process. What it
does not survive is an API restart mid-run.

**Why (b) anyway.** Not because it sounds better — because it is barely more work. The event layer
(typed events, seq assignment, Redis Stream, SSE relay, `Last-Event-ID` replay, heartbeats) is
*identical* under both. The delta is one worker module and one entrypoint. So the trade is not 30%
vs 100% complexity; it is closer to 70% vs 100%, and the extra 30% buys:

* **A stateless API.** You can deploy without killing in-flight 67-second runs. For a system whose
  unit of work is over a minute, "can I ship right now?" is a real operational question, and (a′)
  answers it with "not without losing work."
* **Execution that stops competing with the event loop** serving SSE relays.

It is also the only option under which "worker-based architecture" describes the system rather than
an intention: under (a′) there is no worker.

### arq, not Celery

| | Celery | arq |
|---|---|---|
| Concurrency | sync-first; an async graph means `asyncio.run()` inside each task | async-native, tasks are coroutines |
| Broker | new dependency (kombu, usually RabbitMQ) | Redis, already in the stack |
| Size | large; billiard, prefork | ~2k LOC |
| Windows | prefork unsupported; needs `--pool=solo` or eventlet | plain asyncio |

Atlas is async end to end — `AsyncPostgresSaver`, `redis.asyncio`, async nodes, async provider
clients. Celery would mean a synchronous worker spinning up an event loop to call into an async
graph, plus a second broker beside a Redis we already run. arq is the same shape as the code it
executes.

**Hence arq, not the more familiar Celery.** Described generally, this is an async task queue with
Redis-backed job state; arq is the implementation of that which fits an async stack, where Celery
would fit it badly.

### The Windows trap, spiked before anything was built

`arq.run_worker()` calls `asyncio.run()`, which on Windows selects `ProactorEventLoop`, which psycopg
refuses. The worker builds the Postgres checkpointer, so it would hit this immediately — and the
symptom is `PoolTimeout: pool initialization incomplete`, which points at the database rather than
the event loop. Same trap as Phase 0.

The spike found the seam: `Worker.async_run()` is a coroutine, so `atlas/worker_main.py` builds the
loop itself using the existing `loop_factory`. It also found a second, non-obvious hazard:

```python
# arq's Worker.close()
if not self._handle_signals:
    self.handle_sig(signal.SIGUSR1)
```

Setting `handle_signals=False` looks like the Windows-friendly choice and is exactly backwards —
Windows has no `SIGUSR1`, so shutdown dies with an `AttributeError`. The default `True` is correct,
because arq's `_add_signal_handler` already swallows the `NotImplementedError` Windows raises.

---

## Verified against the installed libraries before designing

| Check | Result |
|---|---|
| `langgraph.config.get_stream_writer` exists in 1.2.10 | yes |
| Writer works inside `Send`-dispatched parallel branches | yes, interleaved in real completion order |
| Writer no-ops under `ainvoke` | yes |
| `stream_mode=["custom","values"]` yields events **and** final state | yes — one execution serves SSE and `/query` |
| Node exceptions propagate out of `astream` | yes |
| Writer **outside any graph** | **raises `RuntimeError`** |

The last one is load-bearing. Phase 2's tests call node functions directly with no graph running, so
an unguarded `emit()` would have broken 126 tests. `emit()` swallows that `RuntimeError`, and a test
asserts the no-op so it stays a deliberate property.

---

## What shipped

| Module | Role |
|---|---|
| `atlas/events.py` | `AgentEvent`, type constants, defensive `emit()`, `node_span` |
| `atlas/eventlog.py` | Redis Stream append / replay / blocking tail, run registration, result storage |
| `atlas/runner.py` | Drives `astream`, assigns `seq`, guarantees exactly one terminal event |
| `atlas/worker.py`, `atlas/worker_main.py` | arq worker; graph built once in `on_startup`; Selector-loop entrypoint |
| `atlas/api/streams.py` | `event_stream` generator and the SSE route |
| `atlas/result.py` | Response building, extracted so the worker and API share one definition |
| `atlas/runtime.py` | `loop_factory` and JSON logging, shared without a circular import |

**Nodes emit payloads; the runner assigns identity.** A node calls `emit(type, node, **fields)` and
knows nothing about `run_id`, `seq`, or Redis.

**Where `seq` comes from.** The obvious reading — concurrent branches, therefore an atomic counter —
is wrong. LangGraph serialises every chunk from every branch into one consuming loop, so the runner
is the sole writer and a plain `+= 1` is correct. No round trip, no lock, and no way to produce a
gap.

**Heartbeats ride the blocking read.** `XREAD BLOCK` timing out *is* the idle tick, so there is no
second timer racing the reader. Heartbeat frames deliberately carry no `id`: one that set an id
would move the client's resume point without delivering an event, so a reconnect straight afterwards
would skip whatever arrived in between.

---

## Verification

```
ruff check .          All checks passed!
mypy --strict atlas   Success: no issues found in 26 source files
pytest -q             161 passed
```

126 → 161 tests. No `# type: ignore` anywhere.

`FakeRedis` gained a real stream implementation rather than a stub, because `XREAD BLOCK` has to
actually block and actually wake or the heartbeat and live-tail tests assert nothing.

### The interleaving test was watched failing

Branch completion order is staggered to be the *reverse* of dispatch order, which is impossible if
the branches ran in series. Forcing the fan-out serial with a lock produced exactly the right
failure:

```
assert ['s0','s1','s2','s3','s4'] == ['s4','s3','s2','s1','s0']
```

Same discipline as `test_concurrency`: the assertion is proven to discriminate, not merely to pass.

---

## Live verification

Worker and API as separate processes, real Postgres, real Redis, real providers.

### Events arrive during the run

```
run_id=223d8b5a…
  t+  0.14s  seq=  1  run_started
  t+  0.16s  seq=  2  node_started       planner
  t+  9.00s  seq=  3  node_completed     planner
  t+  9.01s  seq=  4  fan_out            planner       count=5
  t+  9.03s  seq=  5  node_started       researcher    subtask=1
  t+  9.03s  seq=  6  node_started       researcher    subtask=2
  t+  9.03s  seq=  7  node_started       researcher    subtask=3
  t+  9.04s  seq=  8  node_started       researcher    subtask=4
  t+  9.04s  seq=  9  node_started       researcher    subtask=5
  t+ 24.09s  (heartbeat)
  t+ 25.46s  seq= 10  node_completed     researcher    subtask=1
  t+ 25.49s  seq= 11  node_completed     researcher    subtask=2
  t+ 27.02s  seq= 12  node_completed     researcher    subtask=4
  t+ 28.08s  seq= 13  node_completed     researcher    subtask=3
  t+ 34.47s  seq= 14  node_completed     researcher    subtask=5
  t+ 34.48s  seq= 15  node_started       synthesizer
  t+ 49.56s  (heartbeat)
  t+ 64.56s  (heartbeat)
  t+ 79.60s  (heartbeat)
  t+ 80.53s  seq= 16  node_completed     synthesizer
  t+ 80.54s  seq= 17  iteration_started  critic        iteration=1
  t+ 80.56s  seq= 18  node_started       critic
  t+ 92.02s  seq= 19  node_completed     critic
  t+ 92.02s  seq= 20  critic_verdict     critic        accept faulted=[]
  t+ 92.21s  seq= 21  run_completed                    iterations=1 tokens=56357
```

Not batched at the end. Five branches start together at t+9.03 and **complete out of dispatch
order** — 1, 2, 4, 3, 5 — which is the interleaving, live. Heartbeats fire through the silent
synthesis. `seq` is dense 1..21 and `run_completed` is last.

### Latency: the split

Graph execution and end-to-end `/query` are reported separately, because a regression could come
from streaming or from queue pickup and one number cannot tell them apart.

| Measurement | Value |
|---|---|
| Graph execution, worker-side (`run_started` → terminal event, from event timestamps) | **92.03s** |
| Client-observed stream duration, same run | **92.21s** |
| → SSE relay overhead | **0.18s** |
| `POST /query` end-to-end | **69.09s** |
| Node critical path in that run (planner 8.80 + slowest branch 22.65 + synth 30.26 + critic 6.41) | **68.12s** |
| → queue pickup + result polling | **0.97s** |

**Streaming costs ~0.2s. The queue costs ~1.0s.** Both are attributable and small.

**The 92s run is not a streaming regression, and the evidence is specific.** That run's synthesis
emitted 3,266 tokens at **70.9 tok/s**; Phase 2's measured rate was ~100 tok/s. Every node in the
run was slow — planner 76.5, researchers 78–105, critic 77.1 tok/s — a uniform provider-side
slowdown, not something a per-event `XADD` could cause. Measured directly: **a whole run's 21 events
cost 78ms of Redis, 0.08% of a 92s run.** The two later runs came in at 69.09s and 69.00s against
the 67.24s Phase 2 baseline.

So: wall clock is unchanged within provider variance, and the ~1.2s that *is* attributable to this
phase splits into ~1.0s of queue and ~0.2s of relay. Phase 5's eval suite drives `/query`, so the
number to budget for there is the ~1.0s of queue pickup per trace.

### Killing the client mid-run

```
  t+ 10.23s  seq=  6  node_started  researcher  subtask=2
  >>> CLIENT DISCONNECTING after 6 events <<<
  status immediately after disconnect : running
  status 20s after disconnect         : running

--- reconnect with Last-Event-ID: 6 ---
  t+ 30.27s  seq=  7 … seq= 13     (the backlog, delivered at once)
  t+ 32.28s  seq= 14 … 
  t+ 69.00s  seq= 21  run_completed

events before disconnect : 6   (seq 1..6)
events after reconnect   : 15  (seq 7..21)
terminal event received  : run_completed
```

The run continued with nobody listening, which is what the design claims. Reconnection replayed
exactly seq 7..21 — no overlap, no hole at the join — and the burst at t+30.27 is the backlog that
accumulated during the 20-second gap.

---

## A defect live verification found that the tests did not

**Subscribing immediately after `POST /runs` returned 404.** The endpoint 404s an unknown `run_id`,
and there is a real window between the job being queued and the worker writing `run_started` — so a
client doing the obvious thing hit "unknown run" for a run that was about to start.

The unit tests missed it because the arq double runs jobs **inline**, so events always existed by
the time anything subscribed. A fake that is *too fast* hid a race the real thing has.

Fixed by registering the run at enqueue time, so 404 still means "no such run" rather than "not
yet", with a test that subscribes before any event exists.

---

## Deliberate choices

**No cancellation.** A run outlives its client on purpose. Spend is already bounded by
`MAX_TOKENS_PER_QUERY` and `MAX_CRITIC_ITERATIONS`, so the worst case is one wasted run — cheaper
than a mechanism that races the reconnect window this phase exists to support.

**No token-level streaming.** A 2,750-token answer would become ~2,750 extra events, and "seq is
dense" would then cover thousands of rows that `MAXLEN` eventually trims, making replay inexact.
That belongs in Phase 4 with a UI to justify redesigning the replay contract.

**`run_completed` carries a summary, not the result.** The full sources list and per-call metrics
come from `GET /runs/{run_id}`, once, rather than being replayed on every reconnect.

**A Redis Stream, not pub/sub.** Pub/sub drops anything published while nobody is listening, and
"nobody is listening" is precisely the window reconnection exists to cover.

**Event writes fail open.** Telemetry that can kill the run it observes is worse than no telemetry.
A dropped event leaves a visible hole because `seq` is dense.

---

## Carried into Phase 4

The event stream is the frontend's contract: `fan_out` sizes the branch list, `node_started` /
`node_completed` drive per-branch state, `critic_verdict` shows the quality gate firing, and
`Last-Event-ID` makes a refresh cheap.

`POST /query` is unchanged in shape and still what the eval suite will drive.

## Out of scope

Next.js frontend · token-level `answer_delta` streaming · run cancellation · eval suite · Langfuse ·
pgvector retrieval.
