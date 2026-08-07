"""The provider fallback — the one Phase 0 behaviour that most needs to be true.

These tests exercise the real composition inside ``get_llm`` (constructor kwargs
included), not a reimplementation of it. Both providers are replaced with fakes,
so nothing here touches the network or spends money.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage

from atlas import models
from atlas.models import get_llm


class AlwaysFailsChatModel(FakeListChatModel):
    """Stands in for a provider that is down, overloaded, or rate-limiting."""

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("primary provider unavailable (simulated)")


@pytest.fixture
def captured_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Swap both providers for fakes and record how each was constructed."""
    captured: dict[str, dict[str, Any]] = {}

    def fake_anthropic(**kwargs: Any) -> BaseChatModel:
        captured["primary"] = kwargs
        return AlwaysFailsChatModel(responses=["never returned"])

    def fake_openai(**kwargs: Any) -> BaseChatModel:
        captured["fallback"] = kwargs
        return FakeListChatModel(responses=["fallback answer"])

    monkeypatch.setattr(models, "ChatAnthropic", fake_anthropic)
    monkeypatch.setattr(models, "ChatOpenAI", fake_openai)
    return captured


def test_fallback_fires_when_primary_fails(
    captured_kwargs: dict[str, dict[str, Any]],
) -> None:
    """A failing primary must hand off, and the caller must get a real answer."""
    result = get_llm().invoke("does not matter")
    assert result.text == "fallback answer"


def test_primary_has_zero_retries(captured_kwargs: dict[str, dict[str, Any]]) -> None:
    """max_retries=0 on the primary is what makes the handoff timely.

    Any non-zero value means LangChain works the failing primary through its full
    backoff schedule before ``with_fallbacks`` ever sees the exception — the
    fallback still fires, but far too late to be useful. If this assertion ever
    fails, the fallback has become decorative.
    """
    get_llm()
    assert captured_kwargs["primary"]["max_retries"] == 0


def test_fallback_keeps_its_retries(captured_kwargs: dict[str, dict[str, Any]]) -> None:
    """The fallback is the last line of defence, so it absorbs transient errors."""
    get_llm()
    assert captured_kwargs["fallback"]["max_retries"] == models.FALLBACK_MAX_RETRIES


def test_primary_gets_a_timeout(captured_kwargs: dict[str, dict[str, Any]]) -> None:
    """Without a timeout a hung connection holds the request open indefinitely."""
    get_llm()
    assert captured_kwargs["primary"]["timeout"] == models.PRIMARY_TIMEOUT_SECONDS


def test_configured_models_are_used(captured_kwargs: dict[str, dict[str, Any]]) -> None:
    """Model choice must come from config, never be hardcoded in the provider layer."""
    get_llm()
    assert captured_kwargs["primary"]["model"] == "claude-sonnet-5"
    assert captured_kwargs["fallback"]["model"] == "gpt-4o"


def test_temperature_reaches_both_providers(
    captured_kwargs: dict[str, dict[str, Any]],
) -> None:
    get_llm(temperature=0.7)
    assert captured_kwargs["primary"]["temperature"] == 0.7
    assert captured_kwargs["fallback"]["temperature"] == 0.7
