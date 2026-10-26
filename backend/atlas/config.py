"""Application configuration.

The single source of truth for every runtime setting and secret. Nothing else in
Atlas reads ``os.environ`` directly — everything goes through :func:`get_settings`,
so the full set of knobs is discoverable in one place and validated once at
startup rather than blowing up deep inside a request.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The .env lives at the repository root so a single file serves both this package
# and docker compose (which auto-loads the .env sitting beside docker-compose.yml).
#
# It must be resolved *absolutely*. pydantic-settings interprets a relative
# env_file against the current working directory, and every command in this project
# runs from backend/ — so a bare ".env" would look in backend/, find nothing, and
# report "missing required variables" while the file sits in plain view one level up.
#
#   config.py -> parents[0] = backend/atlas
#             -> parents[1] = backend
#             -> parents[2] = repository root
ENV_FILE: Path = Path(__file__).resolve().parents[2] / ".env"


# How a boolean arrives from the environment, before it is parsed.
_TRUE = {"1", "true", "yes", "on"}


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment and the root .env."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets — no defaults, deliberately. An absent key should stop startup, not
    # surface later as an opaque 401 from a provider. Offline mode, which calls no
    # provider, is the one exception (_no_keys_offline).
    anthropic_api_key: str
    openai_api_key: str
    tavily_api_key: str
    # Offline mode (atlas.offline): the whole product with the models and web
    # search scripted, so it runs, and can be shown, with no API key. Every answer
    # and every call says it was scripted. Off unless asked for.
    atlas_offline: bool = False

    # Models. Changing the model Atlas runs on is a one-line change, here.
    primary_model: str = "claude-sonnet-5"
    fallback_model: str = "gpt-4o"

    # Infrastructure.
    database_url: str
    redis_url: str

    # Runtime.
    log_level: str = "INFO"

    # Token budget for one query. Measured and warned on in Phase 1; enforced in
    # Phase 2, where the critic loop can re-run the fan-out and turn a bounded
    # cost into an unbounded one. Phase 1 spend is already bounded structurally by
    # max_subtasks x max_tool_calls_per_researcher.
    max_tokens_per_query: int = 100_000

    # Research graph.
    max_subtasks: int = 5

    # What Atlas researches from by default (atlas.domains): the compliance
    # copilot, or the general web researcher the first baseline measured. A run
    # can override it in its config.
    atlas_domain: Literal["general", "compliance"] = "compliance"
    # Compliance researchers: provisions read from the regulation library, and web
    # results kept beside them as secondary sources (0 turns the web off).
    library_results: int = 6
    compliance_web_results: int = 3
    # Whether the compliance critic also asks a model if each cited provision says
    # what its sentence says (atlas.library.support). One structured call a round.
    citation_support_check: bool = True

    # Days a conversation is kept after its last question; then the nightly purge
    # deletes it everywhere (atlas.retention). 0 keeps everything.
    retention_days: int = 30
    max_tool_calls_per_researcher: int = 3
    search_results_per_query: int = 5

    # Critic loop. 2, not 5: the first pass catches most errors and returns
    # collapse after two or three, while each extra pass costs a re-research plus
    # a full re-synthesis on top of a 72s baseline.
    max_critic_iterations: int = 2

    # Output ceiling for the synthesizer. Latency is output-driven — measured at
    # ~100 tok/s across every call in both live runs — so this is the single
    # effective lever on wall clock. A backstop, not the mechanism: the concision
    # rule in SYNTHESIZER_SYSTEM does the work and this catches the outliers.
    #
    # 6000, revised up from a planned 2000 across two live runs. The measurements:
    #   Phase 1 prompt, uncapped   4392 out   40.5s
    #   this prompt,   uncapped    2752 out   27.1s   <- the prompt is the mechanism
    #   this prompt,   cap 2000    2000 out   22.2s   <- truncated mid-sentence
    #   this prompt,   cap 3500    3500 out           <- truncated too, on a revision
    # At 2000 and again at 3500 the cap bound on ordinary runs rather than
    # outliers: the answer ended mid-sentence, a URL was cut in half, and the run
    # reported ungrounded_citation for a citation that was never fabricated, only
    # severed. A revision pass carrying critic feedback runs longer than a first
    # pass, so "typical" is not one number. 6000 clears the largest output ever
    # observed here (4392, under the more verbose Phase 1 prompt), which makes this
    # what it was always meant to be — a guard against a runaway, not a shortener.
    # It costs nothing on a normal run, and synthesis_truncated reports any trip.
    synthesis_max_tokens: int = 6000

    # --- Streaming ---------------------------------------------------------
    #
    # Heartbeat comment interval on an idle SSE stream. Proxies and load
    # balancers close connections that go quiet — nginx defaults to 60s — and a
    # run can easily spend 25s inside one synthesis call with nothing to report.
    sse_heartbeat_seconds: int = 15

    # How long a finished run's events and result stay readable. Matches the
    # budget counter's TTL: an abandoned run's event log should not outlive the
    # spend record for the same run.
    event_stream_ttl_seconds: int = 3600

    # Runs the worker will execute concurrently. Each one is five parallel
    # branches of LLM I/O, so this is a fan-out multiplier, not a thread count.
    worker_max_jobs: int = 5

    @property
    def sqlalchemy_url(self) -> str:
        """``database_url`` rewritten for SQLAlchemy's async psycopg3 driver.

        The LangGraph checkpointer talks to psycopg directly and wants a plain
        ``postgresql://`` URL; SQLAlchemy needs the driver named explicitly as
        ``postgresql+psycopg://``. Deriving the second from the first keeps .env
        down to one URL, so there is only ever one value to get right. Idempotent —
        an already-qualified URL passes through untouched.
        """
        prefix = "postgresql://"
        if self.database_url.startswith(prefix):
            return f"postgresql+psycopg://{self.database_url[len(prefix) :]}"
        return self.database_url

    @model_validator(mode="before")
    @classmethod
    def _no_keys_offline(cls, data: object) -> object:
        """Offline mode calls no provider, so it needs no key. Anything else still
        fails as a missing variable, with the message that says how to fix it."""
        if isinstance(data, dict) and str(data.get("atlas_offline", "")).lower() in _TRUE:
            for key in ("anthropic_api_key", "openai_api_key", "tavily_api_key"):
                data.setdefault(key, "")
        return data


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings, or fail loudly with an actionable message.

    Cached because settings are immutable for a process lifetime and are read on
    hot paths. Tests must call ``get_settings.cache_clear()`` when they patch the
    environment, or a previous test's values leak into the next one.
    """
    try:
        return Settings()
    except ValidationError as exc:
        missing = sorted(
            str(error["loc"][0]).upper()
            for error in exc.errors()
            if error["type"] == "missing" and error["loc"]
        )
        if missing:
            raise RuntimeError(
                "Atlas configuration is incomplete.\n"
                f"  Missing required variable(s): {', '.join(missing)}\n"
                f"  Looked for a .env at: {ENV_FILE}\n"
                "  Fix: copy .env.example to .env at the repository root and fill it in.\n"
                "  Or, to try Atlas with no key at all, set ATLAS_OFFLINE=true: every model\n"
                "  is then scripted, and every answer says so."
            ) from exc
        raise RuntimeError(f"Atlas configuration is invalid:\n{exc}") from exc
