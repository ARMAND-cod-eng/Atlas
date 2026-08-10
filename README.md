# Atlas

Atlas is a multi-agent research platform. You ask a question; a planner decomposes it into
independent subtasks, researcher agents investigate them **in parallel** against live web search, a
critic reviews the findings against the sources actually retrieved and can **send specific branches
back** for another pass, and a synthesizer writes the answer with citations. The whole run streams to
the browser as it happens.

It runs on LangGraph behind FastAPI, with an arq worker executing runs, Postgres for checkpointed
state, Redis for the event log and the token budget, and a Next.js frontend that draws the agent
graph live.

**Built and verified against live providers through Phase 4.** The eval suite (Phase 5) and Langfuse
tracing (Phase 6) are not built yet.

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
| Tests | **168** backend · **34** frontend, no network calls, no spend |

---

## Prerequisites

| | |
|---|---|
| Python | 3.12 (pinned in `.python-version`; uv installs it automatically) |
| [uv](https://docs.astral.sh/uv/) | 0.12+ |
| Node | 20+ (developed on 24) |
| Docker Desktop | with the WSL 2 backend |
| API keys | Anthropic, OpenAI, Tavily |

## Run the whole thing

Four terminals. All commands are PowerShell, run from the repository root.

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

A browser `EventSource` sends the `Last-Event-ID` header automatically on reconnect;
the query parameter is for clients like `curl` that do not retry. The same thing works
in the UI — refresh `/runs/{run_id}` mid-run and it rebuilds from the log.

**The synchronous endpoint still exists**, and is what the eval suite will drive:

```powershell
$r = Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType 'application/json' -Body '{"query":"What is retrieval-augmented generation?"}'
$r.answer
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType 'application/json' -Body (@{query='Summarize that in one sentence.'; thread_id=$r.thread_id} | ConvertTo-Json)
```

Passing the returned `thread_id` back confirms the Postgres checkpointer is persisting state.

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
uv run ruff check . ; uv run ruff format . ; uv run mypy --strict atlas ; uv run pytest -q
```

```powershell
cd frontend
npm run typecheck ; npm run lint ; npm test
```

`mypy --strict` with no `# type: ignore` anywhere; TypeScript strict plus
`noUncheckedIndexedAccess`, no `any`, no `eslint-disable`. Neither suite makes a network call or
spends money — every model is a fake, and the frontend's fixtures are real event streams recorded
from the graph by `backend/scripts/capture_fixtures.py`.

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

**Phase 5 — evaluation.** A graded question set, regression runs against known-good answers, and
critic-loop effectiveness measured rather than assumed. Budget ~1.0s of queue pickup per trace.

**Phase 6 — tracing.** Langfuse. `atlas/metrics.py` is deliberately small and shaped to be replaced
by deletion rather than unpicked from everywhere.
