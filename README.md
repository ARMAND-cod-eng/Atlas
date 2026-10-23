# Atlas

[![ci](https://github.com/ARMAND-cod-eng/Atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/ARMAND-cod-eng/Atlas/actions/workflows/ci.yml)

Atlas is a multi-agent research platform. You ask a question; a planner decomposes it into
independent subtasks, researcher agents investigate them **in parallel** against live web search, a
critic reviews the findings against the sources actually retrieved and can **send specific branches
back** for another pass, and a synthesizer writes the answer with citations. The whole run streams to
the browser as it happens.

It runs on LangGraph behind FastAPI, with an arq worker executing runs, Postgres for checkpointed
state, Redis for the event log and the token budget, and a Next.js frontend that draws the agent
graph live.

**Built and verified against live providers through Phase 4.** The Phase 5 eval suite is built and
unit-tested but has not been run yet — see [Evaluation](#evaluation) for exactly where it stands.
Langfuse tracing (Phase 6) is not started.

Phase notes — what was built, the defects found along the way, and how each was verified:

| | |
|---|---|
| [`docs/PHASE-0.md`](docs/PHASE-0.md) | Foundations: config, provider fallback, checkpointing |
| [`docs/PHASE-1.md`](docs/PHASE-1.md) | Planner fan-out, parallel researchers, grounded citations |
| [`docs/PHASE-2.md`](docs/PHASE-2.md) | The critic loop, targeted retry, enforced token budget |
| [`docs/PHASE-3.md`](docs/PHASE-3.md) | Worker, Redis event log, SSE with resumable replay |
| [`docs/PHASE-4.md`](docs/PHASE-4.md) | The frontend and the live agent-graph visualisation |

---

## Architecture

```mermaid
graph LR
    UI[Next.js frontend]
    API[FastAPI]
    Worker[arq worker]

    UI -->|POST /runs| API
    API -->|enqueue| Queue[(Redis queue)]
    Queue --> Worker
    Worker -->|events| Stream[(Redis Stream)]
    Stream -->|SSE + Last-Event-ID| API
    API -.->|live events| UI

    Worker --> Graph
    Graph --> PG[(Postgres<br/>checkpointer)]
    Graph --> Tavily[Tavily search]

    subgraph Graph[LangGraph]
        direction LR
        P[planner] -->|Send x N| R[researcher]
        R --> S[synthesizer]
        S --> C[critic]
        C -->|accept| E([END])
        C -->|revise: faulted branches only| R
        C -->|revise: write-up only| S
    end
```

**The run does not belong to the connection.** The API enqueues and relays; the worker executes. A
client that drops at second 40 of a 67-second run costs nothing — the run finishes, and reconnecting
replays exactly what was missed from the event log. Verified by killing the API mid-run: the run
completed with nothing listening, and the UI converged to the identical final state on restart.

**The critic sends back branches, not the whole graph.** Re-running everything would cost ~72s and
~31k tokens per iteration, and re-planning produces a *different decomposition*, so per-subtask
feedback would have nothing to attach to. Targeted retry costs the slowest faulted branch plus one
synthesis, and preserves subtask identity so guidance reaches the branch that earned it.

### Measured

| | |
|---|---|
| Fan-out is genuinely concurrent | 5 branches: slowest **23.3s**, summed **93.4s** — the fan-out costs the slowest branch, not the sum |
| Typical run | **67–92s** wall clock, dominated by provider tok/s variance rather than by Atlas |
| Streaming overhead | SSE relay **0.18s**; queue pickup **~1.0s**; a whole run's events cost **78ms** of Redis |
| Concision prompt on the synthesizer | 4,392 → 2,752 output tokens, **40.5s → 27.1s** |
| Time to first visual feedback | **1.14s** from submit to the graph on screen |
| Tests | **317** backend · **43** frontend, no network calls, no spend |

---

## Prerequisites

| | |
|---|---|
| Python | 3.12 (pinned in `.python-version`; uv installs it automatically) |
| [uv](https://docs.astral.sh/uv/) | 0.12+ |
| Node | 20+ (developed on 24) |
| Docker Desktop | with the WSL 2 backend |
| API keys | Anthropic, OpenAI, Tavily |

## Quickstart

Docker and three API keys. From the repository root:

```powershell
Copy-Item .env.example .env   # then fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
docker compose --profile app up --build
```

Open <http://127.0.0.1:3000> and ask a question. That starts Postgres, Redis, the API, the worker
and the frontend in containers. To develop with fast reloads instead, run the pieces on the host as
below.

## Run the whole thing

Four terminals, for development. All commands are PowerShell, run from the repository root.

**1. Configure.** One `.env` at the root serves both the backend and Docker Compose.

```powershell
Copy-Item .env.example .env
notepad .env
```

**2. Start infrastructure.**

```powershell
docker compose up -d
```

Confirm both services are healthy and that pgvector was enabled by the init script:

```powershell
docker compose ps
docker compose exec -T postgres psql -U atlas -d atlas -c "\dx vector"
```

> Keep the `-T`. Without it, `docker compose exec` allocates a TTY and psql pipes
> its output through `less` inside the container — leaving your terminal stuck at
> `(END)` with every subsequent keystroke going to the pager instead of the shell.

**3. Install dependencies.**

```powershell
cd backend
uv sync
```

**4. Run the worker** — terminal one. Research runs execute here, not in the API.

```powershell
uv run python -m atlas.worker_main
```

**5. Run the API** — terminal two.

```powershell
uv run python -m atlas --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

**6. Run the frontend** — terminal three.

```powershell
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:3000> and ask a question.

> **Both the worker and the API are required.** With no worker running, `POST /runs`
> still returns a `run_id` and the event stream sits on heartbeats forever, because
> nothing is producing events; `POST /query` waits and returns 504. If a run never
> seems to start, check that the worker terminal is alive.

> **Use `127.0.0.1`, not `localhost` — it is worth two seconds a request.** On
> Windows `localhost` resolves to `::1` before `127.0.0.1`, and uvicorn binds IPv4
> only, so every call waits for the IPv6 attempt to fail first. Measured:
> **2,050 ms via `localhost` against 22 ms via `127.0.0.1`** for the same request.
> The frontend defaults to `127.0.0.1:8000`; override with `NEXT_PUBLIC_API_BASE_URL`
> in `frontend/.env.local`.

## Verifying it works

Health check — both booleans come from live probes, not constants:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

**Watch a run from the terminal.** `POST /runs` returns immediately with a `run_id`;
the event stream shows the graph working. Branches start together and finish out of
order — that interleaving is the fan-out, live.

```powershell
$run = Invoke-RestMethod -Uri http://127.0.0.1:8000/runs -Method Post -ContentType 'application/json' -Body '{"query":"Compare RAG with long-context LLMs for enterprise document QA."}'
curl.exe -N "http://127.0.0.1:8000/runs/$($run.run_id)/events"
```

Kill that `curl` with Ctrl-C partway through and the run keeps going — it belongs to
the worker, not to your connection. Check on it, then resume from where you left off
by passing the last `id:` you saw:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/runs/$($run.run_id)"
curl.exe -N "http://127.0.0.1:8000/runs/$($run.run_id)/events?last_event_id=6"
```

`GET /runs/{run_id}` reports `running`, `completed` with the full result, or `failed` with the
error the run ended on.

A browser `EventSource` sends the `Last-Event-ID` header automatically on reconnect;
the query parameter is for clients like `curl` that do not retry. The same thing works
in the UI — refresh `/runs/{run_id}` mid-run and it rebuilds from the log.

**The synchronous endpoint still exists**, for clients that want one request and one answer:

```powershell
$r = Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType 'application/json' -Body '{"query":"What is retrieval-augmented generation?"}'
$r.answer
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType 'application/json' -Body (@{query='Summarize that in one sentence.'; thread_id=$r.thread_id} | ConvertTo-Json)
```

Passing the returned `thread_id` back continues the conversation: the follow-up is planned against
the previous turn, which confirms the Postgres checkpointer is persisting state. It is still a
separate run, with its own research and its own token budget. A thread runs one question at a time:
a second request while one is in progress gets `409`, because two runs writing one checkpoint would
silently overwrite each other's turn.

**Verify the fallback actually fires.** The behaviour most worth confirming by hand: break
the primary provider and check an answer still comes back via GPT-4o. The key is read by the
**worker**, so set it there:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-deliberately-invalid"
uv run python -m atlas.worker_main
```

An answer means the fallback chain worked. Restart without the override to go back to normal.

## Development

```powershell
cd backend
uv run ruff check . ; uv run ruff format . ; uv run mypy --strict atlas atlas_eval ; uv run pytest -q
```

```powershell
cd frontend
npm run typecheck ; npm run lint ; npm test
```

`mypy --strict` with no `# type: ignore` anywhere; TypeScript strict plus
`noUncheckedIndexedAccess`, no `any`, no `eslint-disable`. Neither suite makes a network call or
spends money — every model is a fake, and the frontend's fixtures are real event streams recorded
from the graph by `backend/scripts/capture_fixtures.py`.

`ci.yml` runs both suites, plus `ruff format --check`, on every push to `main` and every pull
request. It never calls a model.

## Evaluation

`backend/atlas_eval` scores the agent on a 50-query dataset in three categories:

| | What it measures | How |
|---|---|---|
| **Completion** | Does the answer contain each fact on the item's rubric? | Pattern and number checks where possible, an LLM judge for the rest |
| **Grounding** | What fraction of the answer's atomic claims do the retrieved sources support? | The answer is split into claims, and each is checked against the retrieved snippets |
| **Tool use** | Subtask count in the expected range, no redundant searches, retries within the loop's bounds, subtasks genuinely independent | Deterministic checks, plus one judged check for independence |

Grounding here is claim-level. The graph's own `ungrounded_urls` check only confirms that a cited URL
was retrieved; it cannot catch a real URL cited for something it does not say. The suite runs the
real graph in-process against a **frozen corpus** of recorded Tavily results, so two runs are
comparable and CI never touches live search. The judge is GPT-4o at temperature 0: a fixed
temperature is impossible on the Claude models Atlas runs on, and a model should not grade its own
answers. The exception is a run the fallback answered, where GPT-4o is grading itself. Each run's
metrics record which provider answered, so those runs can be told apart.

**Status: dataset reviewed, suite not yet run.** Every one of the 50 items was read and approved by
a person, and nine were corrected first: rubric lines that were wrong, ambiguous, out of date, or
asked for more than the question did. No corpus has been recorded and there is no baseline.
`atlas_eval claim` prints the strongest description the dataset supports, and a test keeps it
honest:

```
50 hand-labeled query traces
```

So this README quotes no eval scores yet. Producing the first ones, from `backend/`:

```powershell
uv run python -m atlas_eval review      # approve or skip each item; resumable
uv run python -m atlas_eval record      # the only online step: live models and Tavily, 3 passes per item
uv run python -m atlas_eval run --name baseline --write-baseline --corpus-date <YYYY-MM-DD>
uv run python -m atlas_eval run --name rerun1          # identical code again: pure noise
uv run python -m atlas_eval noise --runs baseline rerun1
uv run python -m atlas_eval label --run baseline       # human labels for judge calibration
uv run python -m atlas_eval calibrate --run baseline   # judge-human agreement: raw, kappa, prevalence
```

Then commit `atlas_eval/data/corpus/`, `atlas_eval/data/baseline.json` and
`atlas_eval/data/runs/baseline.json`. `noise` reports, per category, how far two runs of identical
code drift apart and so the smallest regression the gate can catch at n=50. A threshold below that
is a promise the gate cannot keep, and gets raised.

The comparison and the chart come from the same run files, never typed in by hand:

```powershell
uv run python -m atlas_eval run --name no-critic --variant no-critic
uv run python -m atlas_eval run --name single-call --variant single-call
uv run python -m atlas_eval report --runs baseline no-critic single-call --out ../docs/RESULTS.md --chart ../docs/img/quality-vs-cost
```

**The CI gate fails closed.** The paid runs live in `eval.yml`, which never fires on a plain push.
Every PR from this repository runs a 12-item smoke that fails if any item crashes. The full 50-item
run, gated against the baseline, runs on PRs that touch prompts, graph code, model config or the
eval itself, on PRs labelled `run-eval`, and on demand. A category fails when it regresses by more
than 2pp *and* its 95% paired-bootstrap CI excludes zero. The gate also refuses to pass when it has
no per-item baseline to pair against. **Until the files above are committed, every eval job
fails.** That is the gate reporting it has nothing to compare against, not a flaky test.

**Comparison runs.** `run --variant no-critic` runs the same graph without the review loop, and
`run --variant single-call` answers each item with one search and Atlas's own synthesizer — same
prompt, same corpus, no planner, researchers or critic. They exist to measure what the full graph
buys, and a baseline can only be written from the production variant.

---

## Things worth knowing

**A component that never fires looks exactly like one that works.** This is the bug class the
project keeps producing, and most of the design exists to make it impossible. A `temperature=0.0`
that 400s meant every answer came from the fallback for a whole session, and everything looked fine.
A `SecretStr` stringified to `'**********'` produced a silent 401 that read as "no results". A
critic whose structured output failed fell open to `accept` and recorded a round byte-for-byte
identical to a real approval. Each was caught by a check built to *discriminate* — a wall-clock
measurement against a forced serial loop, a reducer compared against `operator.add`, a deliberately
poisoned search branch the critic had to reject — and each now reports itself in `metrics.degraded`,
which the UI shows prominently rather than tucking away.

**A thread is a conversation; a run is one question.** Only the conversation carries over between
runs on the same thread. Each run starts with fresh research, critic feedback, metrics and token
budget. That took deliberate work. LangGraph applies graph input through each key's reducer, so
passing `research_results: []` does not clear the previous run's results — it merges into them.
Before this was fixed, a follow-up was synthesised partly from the last question's research, and
the budget counted a whole conversation against a per-question limit. The planner, which runs
once at the start of every run, now resets those keys explicitly. `tests/test_run_isolation.py`
runs two questions on one thread to keep it that way.

**Model selection** is a single config constant, `PRIMARY_MODEL`, read only by `atlas/models.py`.
Nothing else names a model. Override it in `.env` without touching code.

**Provider fallback** sets `max_retries=0` on the primary deliberately. LangChain retries *inside* a
runnable before the exception reaches `with_fallbacks`, so a non-zero value would work an overloaded
provider through its full backoff schedule while the healthy fallback sits idle.
`tests/test_fallback.py` guards it.

**The synthesizer's output cap is a runaway guard, not a length target.** Caps of 2,000 and 3,500
both truncated ordinary answers mid-sentence — severing a URL, which then surfaced as
`ungrounded_citation` for a citation that was never fabricated. The concision *prompt* is what
actually shortens the answer; the cap sits above the largest output ever observed, and a run that
trips it reports `synthesis_truncated`.

**Windows, twice.** psycopg's async driver cannot run on `ProactorEventLoop`, which is what both
uvicorn and arq's CLI select by default — the symptom is `PoolTimeout: pool initialization
incomplete`, which points at the database while the database is fine. `python -m atlas` and
`python -m atlas.worker_main` both build a Selector loop for you. And arq's `handle_signals=False`
looks like the Windows-friendly choice but makes shutdown call `signal.SIGUSR1`, which does not
exist there; the default `True` is correct because arq already swallows the `NotImplementedError`.

**`--reload` does not cover `.env`.** It watches the `backend` package; `.env` lives at the
repository root and is read once at startup. After editing it, stop and restart the process.

## Roadmap

**Phase 5 — evaluation, finishing.** The suite is built; what remains is running it: review the
dataset, record the corpus, commit a baseline, and calibrate the judge against human labels. Then
measure what Atlas claims rather than assume it — the full graph against a single model call with
search at the same token budget, and the critic loop on against off.

**Phase 6 — tracing.** Langfuse. `atlas/metrics.py` is deliberately small and shaped to be replaced
by deletion rather than unpicked from everywhere.
