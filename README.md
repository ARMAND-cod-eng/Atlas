# Atlas

Atlas is a multi-agent research platform: you ask a question, a planner decomposes it into
subtasks, parallel researcher agents investigate them, a critic reviews the findings and can send
work back for another pass, and a synthesizer writes the final answer. It runs on LangGraph behind
a FastAPI service, with Postgres+pgvector for state and retrieval and Redis for caching.

**This repository is currently at Phase 0 — foundations only.** The graph has a single node. The
planner, researchers, critic, synthesizer, SSE streaming, frontend, and eval suite are not built
yet; what exists is the verified scaffolding they will sit on. See
[`docs/PHASE-0.md`](docs/PHASE-0.md) for what was built, the defects found along the way, and how
it was verified.

## Architecture

The target design. Solid edges are built; dashed edges arrive in later phases.

```mermaid
graph TD
    Client[Next.js frontend] -.SSE.-> API
    API[FastAPI service] --> Graph

    subgraph Graph[LangGraph agent graph]
        Start([START]) --> Answer[answer_query]
        Answer --> Done([END])

        Planner[planner] -.-> R1[researcher]
        Planner -.-> R2[researcher]
        Planner -.-> R3[researcher]
        R1 -.-> Critic[critic]
        R2 -.-> Critic
        R3 -.-> Critic
        Critic -.retry.-> Planner
        Critic -.approve.-> Synth[synthesizer]
    end

    Graph --> Postgres[(Postgres + pgvector)]
    API --> Redis[(Redis)]
    Graph -.traces.-> Langfuse[Langfuse]

    style Answer fill:#2d6a4f,color:#fff
    style Planner stroke-dasharray: 5 5
    style Critic stroke-dasharray: 5 5
    style Synth stroke-dasharray: 5 5
```

Phase 0 implements the highlighted path: `START → answer_query → END`, checkpointed to Postgres.
Its purpose is to prove every seam — config, provider fallback, async execution, state persistence,
HTTP surface — before any agent logic depends on them.

## Prerequisites

| | |
|---|---|
| Python | 3.12 (pinned in `.python-version`; uv installs it automatically) |
| [uv](https://docs.astral.sh/uv/) | 0.12+ |
| Docker Desktop | with the WSL 2 backend |
| API keys | Anthropic, OpenAI, Tavily |

## Setup

All commands are PowerShell, run from the repository root.

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
> `-T` prints straight to stdout.

**3. Install dependencies.**

```powershell
cd backend
uv sync
```

**4. Run the API.**

```powershell
uv run python -m atlas --reload
```

Interactive docs at <http://localhost:8000/docs>.

> **Do not launch with bare `uvicorn atlas.api.main:app` on Windows.** It starts,
> then fails after 30 seconds with `PoolTimeout: pool initialization incomplete`.
> uvicorn selects `ProactorEventLoop` on Windows for runs without subprocesses, and
> psycopg's async driver cannot use it — so the checkpointer pool never opens. The
> error names the database, but the database is fine. `python -m atlas` passes the
> right loop factory for you. If you want raw uvicorn, pass it yourself:
>
> ```powershell
> uv run uvicorn atlas.api.main:app --loop atlas.api.main:loop_factory
> ```
>
> Confusingly, bare `uvicorn ... --reload` *does* work — the reloader uses
> subprocesses, for which uvicorn already picks a Selector loop. Don't rely on it.

> **`--reload` does not cover `.env`.** It watches the `backend` package; `.env`
> lives at the repository root and is read once at startup, then cached. After
> editing `.env` — adding an API key, switching `PRIMARY_MODEL` — **stop the server
> and start it again**. A running server keeps serving with the old values, and a
> stale key shows up as a `500` from `/query` with nothing pointing back at config.

## Verifying it works

Health check — both booleans come from live probes, not constants:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Ask a question, then continue the same thread to confirm the checkpointer is persisting state:

```powershell
$r = Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType 'application/json' -Body '{"query":"What is retrieval-augmented generation?"}'
$r.answer
$r.thread_id
```

```powershell
Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType 'application/json' -Body (@{query='Summarize that in one sentence.'; thread_id=$r.thread_id} | ConvertTo-Json)
```

**Verify the fallback actually fires.** This is the behaviour most worth confirming by hand: break
the primary provider and check that an answer still comes back via GPT-4o.

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-deliberately-invalid"
uv run python -m atlas
```

Then from a second terminal:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType 'application/json' -Body '{"query":"Say hello."}'
```

An answer means the fallback chain worked. Restart without the override to go back to normal.

## Development

```powershell
cd backend
uv run ruff check .
uv run ruff format .
uv run mypy --strict atlas
uv run pytest -v
```

The test suite makes no network calls and spends no money — every model is a fake.

## Notes

**Model selection.** The model is a single config constant, `PRIMARY_MODEL`, read only by
`atlas/models.py`; nothing else names a model. Atlas was originally specified against
`claude-sonnet-4-5-20250929` and now defaults to `claude-sonnet-5` — recorded here so the choice
reads as a decision rather than an accident. Override it in `.env` without touching code.

**Provider fallback.** `atlas/models.py` sets `max_retries=0` on the primary model deliberately.
LangChain retries *inside* a runnable before the exception reaches `with_fallbacks`, so a non-zero
value would work an overloaded provider through its full backoff schedule while the healthy
fallback sits idle. See the docstring in that module; `tests/test_fallback.py` guards it.

## Metrics

TODO — token accounting per query, per-agent latency, cache hit rates, and cost attribution.
`MAX_TOKENS_PER_QUERY` already exists in config as the budget these will measure against.

## Evaluation

TODO — the eval suite lands in a later phase: a graded question set, regression runs against
known-good answers, and critic-loop effectiveness measurement.
