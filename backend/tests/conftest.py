"""Shared test fixtures.

Two rules hold across this suite: no test touches the network, and no test spends
money. Every model is a fake, every credential below is fictional.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from atlas.config import get_settings

FAKE_ENV: dict[str, str] = {
    "ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key",
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "TAVILY_API_KEY": "tvly-test-not-a-real-key",
    "DATABASE_URL": "postgresql://atlas:atlas@localhost:5432/atlas",
    "REDIS_URL": "redis://localhost:6379/0",
}


@pytest.fixture(autouse=True)
def fake_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Populate a complete fake environment and isolate the settings cache.

    Environment variables outrank the .env file in pydantic-settings, so these win
    even on a machine with a real .env present. Clearing the lru_cache on both
    sides is what stops one test's settings leaking into the next.
    """
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
