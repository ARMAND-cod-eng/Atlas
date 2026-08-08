"""Application configuration.

The single source of truth for every runtime setting and secret. Nothing else in
Atlas reads ``os.environ`` directly — everything goes through :func:`get_settings`,
so the full set of knobs is discoverable in one place and validated once at
startup rather than blowing up deep inside a request.
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import ValidationError
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


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment and the root .env."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets — no defaults, deliberately. An absent key should stop startup, not
    # surface later as an opaque 401 from a provider.
    anthropic_api_key: str
    openai_api_key: str
    tavily_api_key: str

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
    max_tool_calls_per_researcher: int = 3
    search_results_per_query: int = 5

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
                "  Fix: copy .env.example to .env at the repository root and fill it in."
            ) from exc
        raise RuntimeError(f"Atlas configuration is invalid:\n{exc}") from exc
