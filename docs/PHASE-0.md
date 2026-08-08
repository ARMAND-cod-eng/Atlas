# Phase 0 — Foundations

**Status:** complete · **Completed:** 2026-08-08 · **Commits:** `a7b95d8` … `23b3d97`

## Goal

Build the foundation the multi-agent system will sit on, and prove every seam works before
any agent logic depends on it. Deliberately no planner, researchers, critic, or synthesizer —
just the scaffolding, verified end to end against live infrastructure.

---

## What was built

| Area | Files | What it does |
|---|---|---|
| **Config** | `atlas/config.py` | pydantic-settings; resolves the repo-root `.env` by absolute path; derives the SQLAlchemy URL from a single `DATABASE_URL`; `lru_cache`d; fails with a message naming the missing variable |
| **Provider layer** | `atlas/models.py` | Claude Sonnet 5 primary (`max_retries=0`, 60s timeout) with a GPT-4o fallback via `.with_fallbacks()` |
| **Graph** | `atlas/graph/{state,nodes,builder}.py` | `AtlasState` TypedDict with `add_messages`; async `answer_query` node; `START → answer_query → END` compiled against `AsyncPostgresSaver` over a psycopg pool |
| **API** | `atlas/api/{main,routes}.py`, `atlas/__main__.py` | FastAPI with lifespan-managed graph + Redis; `/health` with live probes; `/query` with thread continuity; structured JSON logging; CORS for `localhost:3000` |
| **DB** | `atlas/db/session.py` | Async engine + session factory, created lazily. Unused in Phase 0; ready for pgvector retrieval |
| **Infra** | `docker-compose.yml`, `docker/init-db/01-vector.sql`, `backend/Dockerfile` | pgvector/pg16 + Redis 7 with healthchecks, a named volume, and a Windows-safe relative mount. The backend image exists but is not yet a compose service |
| **Tests** | `backend/tests/` (7 files) | 27 tests. No network calls, no spend — every model is a fake |

**Stack as resolved:** Python 3.12.13 · langchain-core 1.5.3 · langgraph 1.2.10 ·
langgraph-checkpoint-postgres 3.1.1 · FastAPI 0.141.1 · psycopg 3.3.4 · ruff 0.14.4 ·
mypy 1.19.1. The linters are pinned exactly so a new rule cannot turn CI red without a
reviewed bump.

---

## Decisions

- **`claude-sonnet-5`** as `PRIMARY_MODEL`, rather than the originally specified
  `claude-sonnet-4-5-20250929`. Recorded in the README so the choice reads as a decision.
- **One `DATABASE_URL`** in psycopg form, with the SQLAlchemy variant derived in code —
  one value to keep correct in `.env` instead of two.
- **`db/session.py` built properly** rather than stubbed, so retrieval has a typed entry
  point to grow into.
- **`python -m atlas` entrypoint** instead of bare `uvicorn`, forced by the event loop
  defect below.

---

## Defects found and fixed

Four real defects. Three were invisible to unit tests and only appeared at runtime.

### 1. `.env` resolved against the wrong directory

Caught during plan review, before any code existed. `pyproject.toml` lives in `backend/`
and every command runs from there, so a relative `env_file=".env"` would have resolved
against `backend/` and silently missed the root `.env` — reporting "missing required
variables" with the file in plain view.

Fixed with `Path(__file__).resolve().parents[2]`. A test asserts the resolved path lands on
the repository root, using `docker-compose.yml` as a sentinel, so moving `config.py` fails a
test instead of failing startup.

### 2. psycopg cannot run on Windows' default event loop

Startup died after 30 seconds with `PoolTimeout: pool initialization incomplete` — an error
naming Postgres, while Postgres was perfectly healthy. uvicorn selects `ProactorEventLoop` on
Windows for any run without subprocesses, and psycopg's async driver refuses it:

```
InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode.
```

Setting the event loop *policy* had no effect — uvicorn 0.36+ passes an explicit
`loop_factory` to `asyncio.run` and never consults the policy. Fixed with a Selector factory
wired through uvicorn's `--loop <module>:<attr>` hook.

`--reload` masked this: the reloader uses subprocesses, for which uvicorn already picks a
Selector loop. Working under `--reload` and failing without it is the signature of this bug.

### 3. The primary model failed on every single call

`get_llm(temperature=0.0)` sent an explicit temperature. Current Claude models reject it:

```
400 invalid_request_error: `temperature` is deprecated for this model.
```

Every request 400'd on Anthropic and was answered by GPT-4o instead. **Every answer produced
while this bug was live came from the fallback.** The system looked completely healthy —
`/query` returned good responses throughout.

This is the important lesson from Phase 0: *a fallback that always fires is indistinguishable
from a working system.* The fallback smoke test proved the fallback worked; nothing proved the
primary did.

Fixed by defaulting `temperature` to `None`, which both providers omit from the request. A
regression test asserts no explicit temperature is sent by default.

### 4. Checkpointed history was write-only

`answer_query` invoked the model with only the new question, so persisted messages were
appended every turn and read never. A follow-up on the same `thread_id` replied that it had no
record of any previous question — despite a correct and complete checkpoint. That made
`thread_id` decorative and the README's persistence check misleading: it proved rows were
written, not that they meant anything.

Fixed by sending `state["messages"]` plus the new question.

### Type checking

Nine `mypy --strict` errors were resolved **without a single `# type: ignore`**. The largest
was that mypy needs `plugins = ["pydantic.mypy"]` to understand that `BaseSettings` populates
itself from the environment; without it every required field reads as a missing constructor
argument.

---

## Verification

Confirmed against live infrastructure, not only tests:

```
pgvector 0.8.6                                     enabled by the mounted init script
checkpoints, checkpoint_blobs,                     created by AsyncPostgresSaver.setup()
checkpoint_writes, checkpoint_migrations
/health  → {"status":"ok","db":true,"redis":true}  live probes, not constants
/query   → "noted." then "Teal."                   same thread_id across two calls
providers → anthropic 2, openai 0                  primary answering, fallback idle
history_length → 0 then 2                          history reaching the model
ruff / mypy --strict / pytest                      clean / clean / 27 passed
```

---

## Commits

```
23b3d97  fix: send checkpointed history to the model
2bee8a7  fix: stop sending temperature — it 400s on claude-sonnet-5
ea428f9  docs: note that --reload does not pick up .env changes
0eb95ea  docs: keep -T on docker compose exec to avoid the pager trap
d78ccf6  fix: run the API on a Selector event loop on Windows
a7b95d8  feat: phase 0 foundations — config, model fallback, single-node graph, health checks
```

---

## Deliberately out of scope

Planner, researcher, critic, and synthesizer nodes · the critic loop · SSE streaming · the
Next.js frontend · the eval suite · Langfuse tracing · the pgvector schema and retrieval
layer · the backend service in docker-compose.

---

## Environment notes

Windows-specific traps hit during Phase 0, documented so they cost time only once:

- **Stale `PATH`.** A terminal opened before an installer ran never sees the new entry, and
  new *tabs* inherit the terminal application's frozen copy. Only restarting the whole
  application helps. A PowerShell profile at `$PROFILE` now re-reads `PATH` from the registry
  at shell startup, so new shells self-heal.
- **`docker compose exec` needs `-T`.** Without it, psql pipes output through `less` inside
  the container and leaves the terminal stuck at `(END)`, with later keystrokes going to the
  pager rather than the shell.
- **`--reload` does not watch `.env`.** It watches the `backend` package; `.env` sits at the
  repository root and is read once at startup. Editing it requires a manual restart.
- **Orphaned listeners.** Repeatedly force-killing servers can leave sockets bound to a port
  by processes that no longer exist, shadowing new binds. Use another port; they clear on
  reboot.

---

## Recommended before Phase 1

Two hardening items, both earned during Phase 0:

1. **`/query` should return a 502 with a real message** instead of a bare 500 with an empty
   body. Every diagnosis above would have taken one step instead of ten.
2. **Log a warning when the fallback fires.** Defect 3 hid behind a working fallback for an
   entire session. Something has to say when the primary is dead.

Then the planner fan-out, growing `AtlasState` with the `subtasks` and `research_results`
fields its docstring already anticipates.
