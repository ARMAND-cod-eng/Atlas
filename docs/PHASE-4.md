# Atlas — Phase 4: the Next.js frontend and live agent-graph visualisation

## Context

Phase 3 made runs observable and durable — typed events, a Redis Stream log, an SSE relay with
`Last-Event-ID` replay — and nothing consumed any of it. A user still posted a query and waited a
minute at a blank screen.

This phase builds the client that makes a 67-second multi-agent run legible while it happens. The
thing worth seeing most is the one nobody had ever seen: a critic sending work back to specific
researchers.

---

## Design decision: where graph topology comes from

**Shipped: (b) — `GET /graph/topology`, derived from the compiled LangGraph — with the fan-out node
expanded at runtime.**

**(a) hardcoded in React** — rejected. "It drifts out of date" understates the failure: a hardcoded
graph does not become *wrong*, it becomes wrong **plausibly**. It keeps drawing a sensible topology
that no longer matches what runs, and because node names are how events route to nodes, a rename in
Phase 5 means events arrive for a node the UI does not have — so that node simply never lights up.
One quiet dark box in an otherwise working display. Same shape as the always-firing fallback and the
masked Tavily key.

**(b) the endpoint** — recommended and shipped. `compiled.get_graph()` already returns nodes and
edges with conditional flags, so the endpoint is ~25 lines over it. **The decisive detail is that
`critic -> researcher` is in there**, marked conditional. Because the retry edge is *declared*, the
UI can draw it dormant from the first frame and light it when it fires — so a viewer understands the
loop exists before seeing it happen.

Desync is prevented by a test rather than by discipline: `backend/tests/test_topology.py` pins the
node and edge sets to explicit values. Change the graph and a test fails, loudly, instead of a demo
failing quietly. That test is *supposed* to fail when the graph legitimately changes; the fix is to
update the expected set deliberately, not to relax it into a subset check.

**(c) inferred from events** — rejected. It ties the picture to whatever happened to run, and more
sharply, *edges that
do not fire in a run do not exist in the inference*. On a clean accepted run the critic never
retries, so the retry edge is never inferred — the most interesting structure in the system is
invisible exactly when the system is working well. An inferred graph is a trace of one execution,
not the shape of the system.

### The dynamic subtask count

This defeats all three options equally, because it is not a rendering problem — it is a fact. The
compiled graph has **one** `researcher` node; `Send` turns it into N at runtime, and N does not
exist until the planner has run.

Resolution: topology marks `researcher` with `fan_out: true` (the backend owns the graph, so it is
the right place to know). Before `fan_out` the UI draws one **Researchers** node, idle. On `fan_out`
it expands into one labelled lane per subtask.

**No ghost lanes.** Drawing five placeholders and collapsing to three would imply knowledge we do not
have — the same honesty rule as Part D. Measured live:

```
t=1.14s   Planner:running   Researchers:idle        <- count not yet decided
t=5.51s   Planner:done      R1:idle                 <- fan_out expands the placeholder
t=5.55s   Planner:done      R1:running
```

---

## Layout stability, enforced structurally

Positions come from a memo keyed on a *topology signature* built from node and edge ids only.
Statuses, timings, sources and verdicts are absent from it, so event traffic cannot invalidate the
memo and the canvas cannot reflow while the user is reading it. A test asserts the signature is
constant across every event after `fan_out`, and that it changes exactly twice in a whole run.

A retry changes no topology — the same lanes re-enter `running` — so it costs no relayout, which is
tested explicitly because reflowing at the single most interesting moment of the run would be the
worst possible time.

**One thing the layout had to learn:** feeding the critic's retry edge to dagre made it reverse
edges internally to break the cycle, which shoved the researcher lanes into *two columns* and
destroyed the one visual that matters most — five branches side by side. Back edges are now detected
by depth-first search (derived, not hardcoded, so a different graph later still works), excluded
from ranking, and drawn over the acyclic layout. A feedback edge should look like feedback.

---

## Part D — the long silent stretch

During synthesis exactly two things are known: it started, and it started *n* seconds ago. Both are
shown. There is no spinner, because a spinner implies a third thing.

What fills the silence is **work already done that was previously hidden**. By the time synthesis
begins every researcher has finished and its sources exist — the event stream discarded them and
reported only a `source_count` at the very end. The researcher's `node_completed` now carries
`sources` and `question`, so the user has real, readable content for the 20–45 seconds during which
nothing else can honestly be said. That is not progress indication; it is un-hiding.

---

## Part E — the frontend is not in docker-compose

Compose stays infrastructure-only, which is the convention Phase 0 set and every phase has kept:
Postgres and Redis in containers, application processes native for fast reloads and real debugging.
Next.js dev in a container on Windows means HMR through a WSL2 bind mount over `node_modules` —
slow, prone to missed file events, and buying nothing, since the frontend needs no infrastructure of
its own. Keeping `docker compose up -d` meaning "the things Atlas depends on" also keeps it
unambiguous.

---

## Verification

```
backend    ruff check .          All checks passed!
           mypy --strict atlas   Success: no issues found in 27 source files
           pytest -q             168 passed

frontend   tsc --noEmit          clean
           eslint                clean
           vitest run            34 passed
           next build            succeeds
```

TypeScript strict plus `noUncheckedIndexedAccess` — stricter than asked, because the reducer is a
pile of `Record` lookups and without it every `state.nodes[id]` types as present. No `any`, no
`eslint-disable`.

### Test proportionality — what was worth it, and what was not

**Worth it: the reducer.** It holds every piece of real logic — status transitions, fan-out
expansion, retry counting, seq ordering, gap detection, terminal handling. Pure, synchronous, exact
expected output. Assertions compare whole state maps, because "the node has status running" passes
while four other lanes are wrong.

**Not worth it: component rendering.** Asserting `<GraphCanvas>` renders five nodes tests
`@xyflow/react`. Asserting a node "has a class" is precisely the truthiness trap — it passes whether
the class means running or failed. Skipped deliberately, not overlooked.

**Fixtures are recorded, not written.** `backend/scripts/capture_fixtures.py` runs the real graph,
runner and nodes. Hand-written fixtures would test the reducer against my beliefs about the backend.

And, per your addition: **every recorded happy-path run is a run that did not need the reducer.** The
failure paths are captured too, by running the same production code with controlled inputs —

| fixture | what it pins |
|---|---|
| `partial-failure` | one lane `failed`, four `done`, run **still completes with an answer**, `degraded: ['researcher_failed:s1']` |
| `exhaustion` | critic never accepts → terminates `completed` with `degraded: ['critic_max_iterations']`, answer kept |
| `run-failed` | terminal `run_failed`, error named, **no node left spinning** |
| `retry` | `runs` per lane equals `{s0:1, s1:1, s2:2, s3:1, s4:1}` — targeted retry, as a number |

The partial-failure case is the one that matters most: Phase 1 keeps a run alive by turning a dead
branch into data, so without this the UI would render a clean green run that quietly answered from
four fifths of the research. The answer panel now says so above the answer, not below it.

---

## Live verification

### A real run, end to end

```
t= 1.14s  Planner:running  Researchers:idle   Synthesizer:idle  Critic:idle
t= 5.51s  Planner:done     R1:idle            Synthesizer:idle  Critic:idle
t= 5.55s  Planner:done     R1:running         Synthesizer:idle  Critic:idle
t=16.43s  Planner:done     R1:done            Synthesizer:running
t=26.90s  Planner:done     R1:done            Synthesizer:done  Critic:running
t=32.28s  Planner:done     R1:done            Synthesizer:done  Critic:done
panel: complete · 32.1s · iteration 1 · 11.9k tokens · 5 sources · planner fallback single subtask
```

### Five researchers, concurrently

From the five-branch run, distinct visual states as they occurred:

```
t=18.12s  researching · 5 branches   R1:running R2:running R3:running R4:running R5:running
t=24.00s  researching · 4 branches   R3 done first
t=37.51s  researching · 3 branches   R4 done
t=38.50s  researching · 2 branches   R2 done
t=39.50s  researching · 1 branch     R1 done
t=40.01s  synthesising               R5 done
t=81.50s  reviewing
t=84.50s  complete
```

They finish out of dispatch order — that is the fan-out, visible.

### A forced critic rejection

One branch's first search poisoned with off-topic sources (same technique as Phase 2, patched in the
worker process only — nothing in the repository changed):

```
critic panel : #1 revise · sent back 1
               #2 accept
Researcher 1 : "run 2" badge     <- only this lane
Researchers 2-5 : no badge
backend      : 6 researcher timings for 5 subtasks, iterations 2, degraded []
```

Exactly the faulted lane re-ran. That contrast — one lane re-running, four staying done — *is*
targeted retry, made legible.

### Killing the connection mid-run

The API process was stopped outright 18 seconds into a run, left down for ~12 seconds, then
restarted.

* The run **completed while the API was down** — 13 events, `run_completed` written by the worker
  with nothing listening.
* On restart the browser's `EventSource` retried, resent `Last-Event-ID`, and the UI converged
  **exactly**: iterations 1, 13,216 tokens, 5 sources — matching the event log field for field.
* `seq` dense 1..13, no gap.

A cold load of `/runs/{id}` also rebuilt the entire run from the log, which is the same property
reachable by pressing F5.

### Time to first visual feedback

**1,135 ms** from click to the skeleton graph on screen, with the planner already `running` — the
first status change lands in the same frame.

Two honest caveats. This is Next.js **dev mode**, which compiles the `/runs/[runId]` route on first
navigation; a production build has no such step. And the in-page timing from an earlier attempt was
discarded rather than reported, because the Browser pane was hidden and renderer throttling made the
numbers meaningless — the figure above was taken with the pane visible.

---

## Two defects found by live verification

**1. Every API call cost two seconds.** `GET /graph/topology`, which does no I/O, took 2,090 ms —
while the Redis calls it does not make took 3 ms. A flat ~2s on *every* request:

```
GET http://localhost:8001/graph/topology :  2104.6  2030.8  2031.8  2090.5 ms
GET http://127.0.0.1:8001/graph/topology :    22.6    22.8    19.9    22.0 ms

resolution of 'localhost':  AF_INET6 ('::1', 8001)   <- tried first
                            AF_INET  ('127.0.0.1', 8001)
```

`localhost` resolves to `::1` first and uvicorn binds IPv4 only, so every request waited for the
IPv6 attempt to fail. **2,050 ms against 22 ms.** This is why the run page first appeared to take
many seconds to show anything. Fixed by defaulting the frontend to `127.0.0.1`, allowing both dev
origins in CORS, and using the loopback address throughout the README.

**2. Lanes were labelled with raw UUIDs when the planner degraded.** The planner's fallback path
mints a `uuid4` subtask id, so a degraded run rendered
`Researcher bd6a8d39-fb49-4448-a465-cd52233d3725`. Lanes are now numbered by position; the id
remains the node's identity underneath. Found only because a live run happened to trip the planner
fallback — no fixture had a uuid id, because none of the recorded runs degraded that way.

---

## Carried into Phase 5

The reducer and event contract are stable enough to build on. `POST /query` is untouched and remains
what the eval suite will drive.

The `~1s` of queue pickup measured in Phase 3 is still the per-trace overhead to budget for.

## Out of scope

Eval suite (Phase 5) · Langfuse (Phase 6) · token-level answer streaming · authentication ·
deployment · a runs index or history · mobile layouts beyond not breaking.
