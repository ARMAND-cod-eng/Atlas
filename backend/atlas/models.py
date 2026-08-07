"""LLM provider layer with automatic fallback.

Every model call in Atlas goes through :func:`get_llm`. No other module
constructs a chat model directly, so changing provider, retry policy, or adding a
third link to the fallback chain is a change to this file alone.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from atlas.config import get_settings

# Ceiling on a single primary-model call. Long enough for a substantial answer,
# short enough that a hung connection hands off to the fallback rather than
# holding a request open until the client gives up.
PRIMARY_TIMEOUT_SECONDS = 60.0

# Retry budget for the fallback. It is the last line of defence, so transient
# failures there should be absorbed rather than propagated.
FALLBACK_MAX_RETRIES = 2


def get_llm(temperature: float = 0.0) -> Runnable[LanguageModelInput, BaseMessage]:
    """Return the primary model with the fallback model attached behind it.

    ``max_retries=0`` on the primary is load-bearing, not a stylistic choice.

    LangChain applies retries *inside* the primary runnable, before any exception
    is ever visible to the ``with_fallbacks`` wrapper. With a non-zero value, an
    overloaded Anthropic endpoint gets worked through its entire exponential
    backoff schedule while a perfectly healthy OpenAI fallback sits idle — and by
    the time the failure finally surfaces, the request has already blown its
    latency budget. The fallback technically fires, but so late it is useless.

    Setting it to zero means the first failure propagates immediately and the
    handoff happens fast. Retries still exist; they belong on the fallback, which
    has ``FALLBACK_MAX_RETRIES``. Do not raise this value.
    """
    settings = get_settings()

    # SecretStr, not the bare string: it masks the key in reprs and tracebacks, so
    # a key cannot leak into a log line via an exception dump.
    primary = ChatAnthropic(
        model=settings.primary_model,
        api_key=SecretStr(settings.anthropic_api_key),
        temperature=temperature,
        timeout=PRIMARY_TIMEOUT_SECONDS,
        max_retries=0,  # see docstring — this must stay 0 for the fallback to work
    )
    fallback = ChatOpenAI(
        model=settings.fallback_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=temperature,
        max_retries=FALLBACK_MAX_RETRIES,
    )

    composed: Runnable[LanguageModelInput, BaseMessage] = primary.with_fallbacks([fallback])
    return composed
