# Phase 1 — Planner Fan-Out and Parallel Researchers

**Status:** complete · **Completed:** 2026-08-09 · **Commits:** `caf8794` … `3562a87`

## Goal

Replace Phase 0's single `answer_query` node with the real shape of the system: a planner that
decomposes a question into independent subtasks, researchers that run those subtasks in parallel
against web search, and a synthesizer that merges the findings with citations. Plus pay off the
observability debt Phase 0 identified, so the next defect takes one diagnostic step instead of ten.

Critic loop, SSE streaming, frontend, eval suite, and Langfuse remain out of scope.

---

## Part A — hardening carried over from Phase 0

| Gap | Fix |
|---|---|
| `/query` returned a bare 500 with an empty body | `atlas/errors.py` adds `AtlasError`, `ProviderError`, `SearchError`. Provider and search failures return **502** naming the provider, the node, and the detail. Unhandled exceptions return 500 with a body and a logged traceback. |
| A broken primary hid behind a working fallback for a whole session | `FallbackWatcher` logs a WARNING on every model failure, carrying the exception that caused the handoff. Every response's model name is recorded, so *which* provider answered is visible per call. |
| No cost or latency visibility | `atlas/metrics.py`: `LLMCall`, `NodeTiming`, `RunMetrics`, one reducer, one context manager. Deliberately small so Langfuse can replace it by deletion. |

`models.py` was refactored into private `_build_primary` / `_build_fallback` builders plus three
public composers. This was **required, not cosmetic** — see Constraint 1 below.

---

## Part B — the research graph

```
START → planner → [fan_out] → researcher (parallel) → synthesizer → END
```

| Node | Behaviour |
|---|---|
| `planner` | Structured output produces independent subtasks, truncated to `MAX_SUBTASKS`. On failure it degrades to a single subtask carrying the original query rather than failing the run ahead of the whole fan-out. |
| `fan_out` | Returns one `Send` per subtask — this is what creates the parallelism. An empty subtask list routes straight to the synthesizer instead of returning an empty `Send` list, which would leave the synthesizer with no results and no explanation. |
| `researcher` | Fixed search-then-summarize. Its entire body is wrapped: a failure becomes a `ResearchResult` with `error` set, never an exception. `MAX_TOOL_CALLS_PER_RESEARCHER` bounds search retries on empty results. |
| `synthesizer` | Merges with citations, names failed subtasks, and states plainly when no research happened rather than answering from model priors. |

State carries `Subtask`, `ResearchResult`, and `Source` as Pydantic models. Two keys need reducers
because every parallel branch writes them: `research_results` uses `operator.add` and `run_metrics`
uses `merge_run_metrics`. Without those, concurrent writes raise `InvalidUpdateError` instead of
merging.

`/query` now returns `{answer, thread_id, subtasks[], sources[], metrics}` — additive, so anything
reading `.answer` still works. `metrics.degraded` means a partial run is visible to the client
without reading server logs.

### Design decision: fixed pipeline, not a ReAct agent

The planner has already done the adaptive work — each subtask arrives as a specific, self-contained
question, which is precisely the case where iteration is *least* needed. Against that modest upside:
cost and latency variance multiply across five concurrent branches; tool-call parsing and loop
non-termination would add failure modes to the one node whose resilience is the headline
requirement; and it would obscure what this phase is actually proving.

`MAX_TOOL_CALLS_PER_RESEARCHER` bounds **search attempts**, not agency. The upgrade path is open:
swapping the body of `researcher()` for a ReAct loop changes nothing about the graph shape, the
state contract, or any test here. That is a decision to make with eval data, not on speculation.

---

## Constraints discovered

Verified against the installed packages, not assumed.

**1. `get_llm()` cannot do structured output or tool binding.** It returns a
`RunnableWithFallbacks`, which has neither `with_structured_output` nor `bind_tools` — both live on
`BaseChatModel`. The transformation must be applied to each model *before* composing the fallback,
or the fallback is silently lost.

**2. A `Send` payload becomes the target node's entire input**, not a patch over `AtlasState`. The
researcher therefore takes its own `ResearcherInput`; its return merges back through the reducers.

**3. `with_structured_output` returns `dict | BaseModel`.** Narrowed at runtime with
`model_validate` rather than a cast, which also re-validates the model's output.

---

## Defects found — all four by live verification, none by tests

Every one was a **failure wearing the costume of success**. That is the same shape as the Phase 0
defect, and the reason this project keeps live verification as a required step.

### 1. The Tavily credential was sent as a mask

Every researcher returned `Error 401: Unauthorized` against a valid key.

`build_search_tool` passed `SecretStr(settings.tavily_api_key)` into `TavilySearchAPIWrapper`. That
wrapper's `mode="before"` validator runs `get_from_dict_or_env`, which **stringifies** whatever it
is handed — and `str(SecretStr("tvly-…"))` is `'**********'`. The wrapper stored the mask and sent
ten asterisks as the credential.

```
settings.tavily_api_key         : len=41 prefix='tvly-dev'
tool.api_wrapper.tavily_api_key : len=10 prefix='********'   ← the mask
api_wrapper=SecretStr(...)  → 401 Unauthorized
tavily_api_key=<plain str>  → 3 results
```

`TavilySearch.__init__` accepts `tavily_api_key` directly and builds the wrapper itself
(`tavily_search.py:330-340`), so that is what we use, with a plain `str`. No `os.environ` mutation —
that would be a hidden global mutation.

**Why it survived review:** the original probe asserted the wrapper key was *truthy*, and
`'**********'` is truthy. The tests now assert the key the client holds **equals** the configured
value, plus a named check that it is not the mask. Truthiness assertions on credentials are
worthless.

### 2. Tavily reports failures in-band, and they read as "nothing found"

`TavilySearch` catches its own transport and API failures and returns `{"error": Exception(...)}`
rather than raising, so the `try/except` around `ainvoke` never fired. The payload has no `results`
key, so the normalizer read it as an honest empty search. A 401 produced a completed run in which
every researcher reported zero sources and nothing was logged.

`_raise_if_error` now converts an in-band error into `SearchError` *before* normalization. An error
arriving alongside populated results also raises — a partial failure must not be read as success.

### 3. Structured-output calls reported provider `unknown`

`with_structured_output` returns only the parsed model, which carries no response metadata, so the
planner's call could not be attributed. A planner served by the fallback would have been invisible —
exactly the blind spot Part A exists to close. Fixed with `include_raw=True`; metrics read the raw
message, the caller gets the parsed model.

### 4. The planner discarded a valid plan

Observed live: `claude-sonnet-5` returned `PlannerOutput` with `subtasks` double-encoded as a JSON
string instead of an array. Strict validation rejected it and collapsed a good five-subtask plan
into the single-subtask fallback — losing the entire fan-out while the correct content sat in the
payload. Both structured-output methods produce this intermittently (3/3 clean on a short prompt,
failing on the production one), so it is handled at the schema with a before-validator rather than
by picking a method and hoping. A genuinely unparseable value still fails loudly.

`call_structured` now logs the raw payload at WARNING before discarding a response, because this one
was found only by chance while reading logs.

---

## Testing: why `test_concurrency` is the load-bearing test

`test_parallel_merge` proves results merge without loss — and **a fully sequential implementation
would pass it identically**. It verifies the outcome, not the mechanism.

Parallelism is the feature of this phase, so it gets a test that fails when parallelism breaks: four
fake researchers each sleeping 0.2s, asserting total wall-clock under 0.5s against a 0.8s sequential
floor. Measured to confirm the assertion actually discriminates:

| | |
|---|---|
| Real fan-out | **0.236s** — four branches overlapping |
| Forced serial loop | **0.853s** |
| Asserted ceiling | 0.500s, cleanly between |

81 tests total. No network calls, no spend — Tavily is mocked entirely and every model is a fake.

---

## Verification

```
ruff check .          All checks passed!
mypy --strict atlas   Success: no issues found in 17 source files
pytest -q             81 passed
```

No `# type: ignore` anywhere. Two places wanted one and got a real fix: Starlette types exception
handlers as taking `Exception`, so ours accept `Exception` and narrow with `isinstance`; and the
`implicit_reexport` override added for `TavilySearchAPIWrapper` was **removed** once defect 1 made
that import unnecessary.

### Live run — 5 branches, real citations

```
total wall-clock       : 71.93s
slowest researcher     : 23.33s   (branches: 12.26  23.33  16.52  22.09  19.24)
summed researcher time : 93.44s   ← what sequential would have cost
planner 14.63s + slowest 23.33s + synthesizer 33.74s = 71.70s ≈ wall clock

sources 25 · distinct URLs cited 18 · ungrounded 0 · degraded (none)
every llm_call → anthropic/claude-sonnet-5   tokens in=20563 out=11033
```

That arithmetic is the parallelism proof: the fan-out costs the **slowest branch**, not the sum.

Citation grounding is enforced, not merely prompted. `ungrounded_urls` compares every URL in the
answer against the sources actually retrieved and records `ungrounded_citation` when they diverge. A
prompt is an instruction; this is a check. Fabricated citations are the worst failure mode a
research system has, because they look verifiable.

---

## Deferred deliberately

**`MAX_TOKENS_PER_QUERY` is measured, not enforced.** Phase 1 spend is already bounded structurally
by `MAX_SUBTASKS × MAX_TOOL_CALLS_PER_RESEARCHER`. Enforcement is also awkward here: parallel
branches read the same pre-fan-out state snapshot, so a counter in graph state cannot see sibling
spend mid-flight, and half-enforcing is worse than none. It lands in Phase 2 with the critic loop,
which is what actually makes spend unbounded. Totals accumulate today and a breach logs a WARNING
plus `budget_exceeded` in `degraded`.

## Out of scope

Critic loop and its routing · SSE streaming · Next.js frontend · eval suite · Langfuse (this phase's
`RunMetrics` is its stand-in and is shaped to be replaced) · pgvector retrieval · the backend compose
service.

---

## Carried into Phase 2

**The synthesizer is the next bottleneck.** It took 33.7s on 8,118 input tokens — nearly half the
wall clock — and grows linearly with branch count. The critic loop re-runs synthesis, so that is
where latency and the token budget bite first, and it is the natural place to finally enforce
`MAX_TOKENS_PER_QUERY`.

**Two lessons worth carrying forward.** A truthiness assertion on a credential proves nothing —
compare against the configured value. And any dependency that reports failures in its return value
rather than by raising needs an explicit check, because the default reading will be "success with
no data".
