# Atlas — Phase 2: the critic loop and cost enforcement

## Goal

Close the core architecture with the quality gate that can send work back, and add the cost controls
that a loop makes mandatory. Phase 1 ended at a bounded cost: five branches, one pass, ~72s. A naive
critic loop turns that into ~3.5 minutes and ~95k tokens per query. Every decision below is about
buying quality without paying that.

---

## What shipped

| Node / module | Behaviour |
|---|---|
| `critic` | Structured verdict per subtask, judged against **the sources actually retrieved**, not against the model's own knowledge. Also the budget checkpoint — it is the only node from which a new iteration can begin. |
| `route_after_critic` | Derives routing from verdict content: accept → END, faulted subtasks → `Send` to *those researchers only*, synthesis feedback alone → re-write, exhaustion → END with a note. |
| `merge_research_results` | Replaces `operator.add` on `research_results`. Upserts by `subtask_id` so a retried branch **replaces** its result instead of appearing twice. |
| `atlas/budget.py` | Per-thread token counter in Redis, `INCRBY` after every model call, TTL 3600. Fails open. |
| synthesizer | Concision rule in the prompt, `max_tokens` backstop, and a `synthesis_truncated` flag when the backstop trips. |

`/query` gains `iterations` and `metrics.critic_rounds`, so the quality gate is visible from the
client: one round recording `accept` means it reviewed and approved, which is a different thing from
it never having run.

---

## Design decision: what re-runs on rejection

**(b) targeted branch retry, with (c) synthesis-only as a distinct verdict.**

**(a) re-run the entire graph** — rejected. Beyond ~72s and ~31k tokens per iteration, it breaks the
feedback loop's coherence: re-planning produces a *different decomposition*, so iteration 1's
`subtask_id`s do not exist in iteration 2 and per-subtask feedback has nothing to attach to. You
would be re-deciding a decomposition the critic never faulted, and discarding the only mechanism
that lets a retried researcher learn from the critique.

**(b) targeted branch retry** — the recommendation, and what the live run measured: the slowest
faulted branch plus synthesis, with typically one branch faulted rather than five. It mirrors how
review actually works, and it preserves subtask identity so guidance maps onto exactly the branch
that earned it.

**(c) synthesis-only** — kept as a distinct verdict rather than folded into (b). "The research is
sound but the write-up missed something" is real and common, and it costs one synthesis call with
zero searches. Collapsing it into (b) would re-run searches to fix a writing problem.

---

## The load-bearing change: `research_results` can no longer be `operator.add`

Appending was correct exactly as long as every subtask ran once. Once a faulted subtask can be
re-researched, appending produces **two** results for one `subtask_id` — the synthesizer then sees
one subtask twice with contradictory findings, and `_dedupe_sources` keeps whichever URL it met
first, which may be the stale one. Nothing raises. The answer is quietly built on rejected research.

The reducer is correct in both regimes, which is the point. On the initial fan-out every id is new,
so every result appends and Phase 1's merge and concurrency tests keep their meaning. On a retry the
id already exists, so the result replaces in place and ordering does not shuffle between iterations.

**Proven by negative control, not by passing.** `test_targeted_retry` runs `operator.add` and
`merge_research_results` over the *same* retry input in one test body and asserts **6 against 5**.
Reverting the reducer was then done for real, and the graph test failed with `assert 6 == 5` — which
also exposed a second test that was *not* discriminating: indexing results by `subtask_id` silently
keeps the last duplicate, so it passed against both reducers. It now compares the whole collection.

---

## Verification

```
ruff check .          All checks passed!
mypy --strict atlas   Success: no issues found in 18 source files
pytest -q             126 passed
```

81 → 126 tests. No `# type: ignore` anywhere.

Every model is faked and every credential fictional, as before. New in this phase: the Redis client
behind the budget counter is replaced by an in-memory fake in an **autouse** fixture, because the
critic checks the budget on every graph run and without it any graph test would open a socket to
localhost:6379.

---

## Live run 1 — the synthesis output cap, measured

Three variants of the same query in one process, so the only differences are the two variables under
test. Variant A reproduces the Phase 1 synthesizer.

| variant | prompt | cap | synthesis | output tokens | wall clock |
|---|---|---|---|---|---|
| A | Phase 1 | none | **40.48s** | 4,392 | 84.66s |
| B | + concision rule | none | **27.07s** | 2,752 | 64.67s |
| C | + concision rule | 2,000 | **22.18s** | 2,000 | 67.24s |

**The prompt is the mechanism; the cap is not.** A → B saves **13.41s of synthesis (33%)**, which is
**15.8% of A's 84.66s wall clock**. The plan projected ~19% from a 2,000-token cap. The measured
result is 15.8%, and it comes from the prompt alone.

C's extra 4.89s is not a saving — it is truncation. Output stopped at exactly 2,000, the answer
ended mid-sentence, and a URL was cut in half, so the run reported `ungrounded_citation` for a
citation that was never fabricated, only severed. Raising the cap to 3,500 truncated too, on a
revision pass carrying critic feedback. A revision runs longer than a first pass, so "typical" is not
one number.

**Shipped: `SYNTHESIS_MAX_TOKENS=6000`**, above the largest output ever observed here (4,392, under
the more verbose Phase 1 prompt). That makes it what it was always meant to be — a guard against a
runaway, not a shortener. It costs nothing on a normal run, and any trip is reported.

The plan asked that the number be reported whatever it turned out to be. It is 15.8%, from the half
of the change that was expected to be the smaller half.

---

## Live run 2 — the critic in both directions

Reporting one direction proves nothing. A critic that always accepts looks *perfect* on every
metric: fast, cheap, one iteration, no degraded flags, 100% accept rate.

**Sound research.** Four real runs across the latency variants and the budget check. Every one
returned `overall == "accept"` on pass 1, one iteration, no critic-related degradation.

**Deliberately weak research.** One branch's first search was poisoned with real but entirely
off-topic sources (sourdough baking). Everything else was real — real planner, real Tavily, real
critic. The retry received genuine sources.

```
[poisoned] 'What is the current state of empirical research comparing RAG versus
            long-context language models on QA accuracy...' -> off-topic sources
[retry   ] same query -> real sources

wall clock      : 148.10s
iterations      : 2
subtasks        : 5
research_results: 5          <- not 6
researcher calls: 6          <- five, plus exactly one targeted retry
synthesis calls : 2
degraded        : []

critic rounds:
  iteration 1: revise  faulted=['1']
  iteration 2: accept  faulted=[]

pass 1, subtask 1: [revise]
  reason  : "The retrieval system returned completely irrelevant sourdough baking
             sources for the accuracy comparison question..."
  the other four subtasks: [accept]

branches still holding poisoned sources: []
```

It faulted the poisoned branch and only the poisoned branch, named the actual reason, accepted the
four sound ones, and the loop **terminated by acceptance** rather than by exhaustion. That is the
whole loop working end to end with real models deciding each step.

---

## Live run 3 — budget breach

`MAX_TOKENS_PER_QUERY=500`, through the real HTTP route against Postgres and Redis:

```
HTTP status  : 200
iterations   : 1
degraded     : [planner_fallback_single_subtask, budget_exceeded]
critic rounds: 1=accept
answer chars : 3767
```

200 with a usable answer, not an error. Overspending is a degraded result; a 5xx would throw away an
answer the user already paid for. (The planner fallback in that run was an unrelated transient — and
it is visible, which is the point of recording it.)

---

## Three defects found by live verification

None of these were visible in 120 passing tests.

**1. A broken critic was indistinguishable from an approving one.** Variant A's critic call failed
structured-output validation. `_review` fell open to accept — correct behaviour, an answer already
existed — but recorded `critic_rounds: [(1, 'accept')]` with `degraded == []`. Byte-for-byte what a
genuine approval looks like. The answer went out unreviewed and nothing said so. Fixed with
`critic_unavailable`, and the test now asserts the flag rather than just the accept.

**2. The output cap truncated ordinary answers.** Covered above. The critic itself caught it on one
run — *"The draft answer is cut off mid-sentence in the Latency section (ends at
'https://dev.to/pooyagolchian/rag-pipelines-in-production-vector-')"* — which is also what confirmed
the `ungrounded_citation` flags were severed URLs rather than fabricated ones. Fixed by sizing the
cap as a backstop and adding `synthesis_truncated` so a badly-set ceiling announces itself.

**3. One breach reported twice.** The synthesizer's in-state estimate and the critic's Redis check
both write `budget_exceeded`. Both are worth keeping — the critic's is authoritative, the
synthesizer's still works when Redis is down — but a duplicated flag tells the client nothing.
Fixed, and the test was watched failing with `['budget_exceeded', 'budget_exceeded']` before the fix
went in.

---

## Deliberate choices

**The budget counter is in Redis, not graph state.** Every parallel branch reads the same
pre-fan-out snapshot, so a counter in state cannot see sibling spend while the fan-out is in flight:
five researchers would each believe they were the only one spending.

**It fails open.** Any Redis error logs a warning, records `budget_check_unavailable`, and lets the
run proceed. A dead cache breaking every query is far worse than an overspent one, and the run stays
bounded by `MAX_CRITIC_ITERATIONS` and `MAX_SUBTASKS` regardless.

**Mid-branch abort is not implemented.** A bounded overshoot of one branch is acceptable and far
simpler than tearing down work in flight.

**`recursion_limit` is explicit**, derived from `MAX_CRITIC_ITERATIONS` rather than left at
LangGraph's default 25, so a routing bug surfaces as a clear error instead of 25 silent supersteps.

**"Revise" with nothing actionable exits.** A verdict naming no subtask and no synthesis problem is
not an instruction; looping on it would burn a full iteration to produce the same answer. Recorded as
`critic_revise_not_actionable`.

---

## Carried into Phase 3

**The critic's input is the largest single prompt in the system** — 22–25k input tokens per call,
because it receives every subtask's findings *plus* every retrieved snippet. That is what makes the
review grounded, and it is cheap in latency (~11s, input is nearly free) but not in cost. It is now
the dominant term in tokens per query.

**A rejection roughly doubles wall clock**: 64.67s accepted on pass 1 against 148.10s for
reject-retry-accept. The retry itself is cheap; the second full synthesis is not.

## Out of scope

SSE streaming · Next.js frontend · eval suite · Langfuse · pgvector retrieval · the backend compose
service.
