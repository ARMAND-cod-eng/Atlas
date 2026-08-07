"""Configuration loading, URL derivation, and failure messaging."""

from __future__ import annotations

import pytest

from atlas.config import ENV_FILE, Settings, get_settings


def test_loads_from_environment() -> None:
    settings = get_settings()
    assert settings.anthropic_api_key == "sk-ant-test-not-a-real-key"
    assert settings.database_url == "postgresql://atlas:atlas@localhost:5432/atlas"


def test_defaults_applied() -> None:
    settings = get_settings()
    assert settings.primary_model == "claude-sonnet-5"
    assert settings.fallback_model == "gpt-4o"
    assert settings.log_level == "INFO"
    assert settings.max_tokens_per_query == 100_000


def test_settings_are_cached() -> None:
    assert get_settings() is get_settings()


def test_sqlalchemy_url_adds_driver() -> None:
    settings = get_settings()
    assert settings.sqlalchemy_url == "postgresql+psycopg://atlas:atlas@localhost:5432/atlas"


def test_sqlalchemy_url_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-qualified URL must pass through untouched, not get double-prefixed."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://atlas:atlas@localhost:5432/atlas")
    get_settings.cache_clear()
    assert get_settings().sqlalchemy_url == "postgresql+psycopg://atlas:atlas@localhost:5432/atlas"


def test_env_file_resolves_to_repository_root() -> None:
    """Guard the parents[2] arithmetic in config.py.

    A relative env_file would resolve against backend/ (where every command runs)
    and silently find nothing. This asserts the absolute path lands on the repo
    root, using docker-compose.yml as a sentinel that exists only there — so if
    config.py is ever moved, this fails loudly instead of the app failing at
    startup with a confusing "missing variables" error.
    """
    assert ENV_FILE.name == ".env"
    assert (ENV_FILE.parent / "docker-compose.yml").is_file()
    assert (ENV_FILE.parent / "backend" / "pyproject.toml").is_file()


def test_missing_required_variable_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key must produce an actionable message, not a raw pydantic dump."""
    # Disable the .env file for this test so the assertion holds on a machine that
    # has a real .env sitting at the repository root.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError) as excinfo:
        get_settings()

    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert ".env.example" in message
