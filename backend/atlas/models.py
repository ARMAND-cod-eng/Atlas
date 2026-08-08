"""LLM provider layer with automatic fallback.

Every model call in Atlas goes through this module. No other module constructs a
chat model, so changing provider, retry policy, or adding a third link to the
fallback chain is a change to this file alone.

Structured output and tool binding get their own composers rather than being
applied to :func:`get_llm`'s result. That is not a style choice —
``RunnableWithFallbacks`` has neither ``with_structured_output`` nor
``bind_tools``; both live on ``BaseChatModel``. The transformation therefore has
to be applied to each model *before* the fallback is composed, or the fallback is
silently lost.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from atlas.config import get_settings
from atlas.errors import ProviderError
from atlas.metrics import LLMCall, RunMetrics

logger = logging.getLogger(__name__)

# Ceiling on a single primary-model call. Long enough for a substantial answer,
# short enough that a hung connection hands off rather than holding a request
# open until the client gives up.
PRIMARY_TIMEOUT_SECONDS = 60.0

# Retry budget for the fallback. It is the last line of defence, so transient
# failures there should be absorbed rather than propagated.
FALLBACK_MAX_RETRIES = 2


class FallbackWatcher(BaseCallbackHandler):
    """Logs a WARNING whenever a model call fails.

    In Phase 0 a completely broken primary hid behind a working fallback for an
    entire session: every response was correct, and every one came from the
    fallback. Nothing said so. This is the thing that says so.

    ``on_llm_error`` fires for the primary's failure before the fallback is
    attempted, so the warning names the exception that caused the handoff.
    """

    # **kwargs is object rather than Any: the base class passes run ids and tags
    # we do not read, and object keeps them out of the type system entirely.
    def on_llm_error(self, error: BaseException, **kwargs: object) -> None:
        logger.warning(
            "model call failed; falling back",
            extra={
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            },
        )


def _build_primary(temperature: float | None = None) -> ChatAnthropic:
    """The primary model.

    ``max_retries=0`` is load-bearing. LangChain applies retries *inside* the
    runnable, before any exception reaches the fallback wrapper — a non-zero
    value works an overloaded endpoint through its full backoff schedule while a
    healthy fallback sits idle, so the handoff happens far too late to be useful.
    Retries belong on the fallback, which has them.

    ``temperature`` defaults to ``None``, meaning the field is not sent at all.
    Current Claude models reject an explicit temperature with
    ``400: `temperature` is deprecated for this model.`` — a default of 0.0 made
    every primary call fail, invisibly, because the fallback answered.
    """
    settings = get_settings()
    return ChatAnthropic(
        model=settings.primary_model,
        api_key=SecretStr(settings.anthropic_api_key),
        temperature=temperature,
        timeout=PRIMARY_TIMEOUT_SECONDS,
        max_retries=0,  # see docstring — must stay 0 for the fallback to work
        callbacks=[FallbackWatcher()],
    )


def _build_fallback(temperature: float | None = None) -> ChatOpenAI:
    """The fallback model, used only when the primary fails."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.fallback_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=temperature,
        max_retries=FALLBACK_MAX_RETRIES,
        callbacks=[FallbackWatcher()],
    )


def get_llm(temperature: float | None = None) -> Runnable[LanguageModelInput, BaseMessage]:
    """The primary model with the fallback attached behind it."""
    composed: Runnable[LanguageModelInput, BaseMessage] = _build_primary(
        temperature
    ).with_fallbacks([_build_fallback(temperature)])
    return composed


def get_structured_llm(
    schema: type[BaseModel],
    temperature: float | None = None,
) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
    """A fallback-backed model constrained to ``schema``.

    ``with_structured_output`` is applied to each model *before* composing, since
    ``RunnableWithFallbacks`` does not have the method. Calling it on the result
    of :func:`get_llm` raises ``AttributeError``.

    The declared return is a union because that is what LangChain declares.
    :func:`call_structured` narrows it at runtime.
    """
    primary = _build_primary(temperature).with_structured_output(schema)
    fallback = _build_fallback(temperature).with_structured_output(schema)
    composed: Runnable[LanguageModelInput, dict[str, Any] | BaseModel] = primary.with_fallbacks(
        [fallback]
    )
    return composed


def _provider_for(model_name: str) -> str:
    """Attribute a response to a provider from the model name it reports."""
    lowered = model_name.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "unknown"


def _record(
    metrics: RunMetrics,
    node: str,
    response: object,
    duration_s: float,
) -> None:
    """Append an :class:`LLMCall` describing what just answered.

    Reads the model name off the response rather than off config, because the
    two differ precisely when it matters — when the fallback answered.
    """
    meta = getattr(response, "response_metadata", None)
    model = "unknown"
    if isinstance(meta, Mapping):
        raw_model = meta.get("model_name") or meta.get("model")
        if isinstance(raw_model, str):
            model = raw_model

    # usage_metadata lives on AIMessage, not BaseMessage, and is absent entirely
    # on structured-output responses. Missing counts are reported as None rather
    # than zero, so "not measured" stays distinguishable from "measured as zero".
    raw_usage = getattr(response, "usage_metadata", None)
    input_tokens: int | None = None
    output_tokens: int | None = None
    if isinstance(raw_usage, Mapping):
        got_in = raw_usage.get("input_tokens")
        got_out = raw_usage.get("output_tokens")
        input_tokens = got_in if isinstance(got_in, int) else None
        output_tokens = got_out if isinstance(got_out, int) else None

    metrics.llm_calls.append(
        LLMCall(
            node=node,
            provider=_provider_for(model),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_s=duration_s,
        )
    )


async def call_model(
    node: str,
    messages: LanguageModelInput,
    metrics: RunMetrics,
    temperature: float | None = None,
) -> BaseMessage:
    """Invoke the composed model, record metrics, translate failures.

    Nodes call this rather than ``get_llm().ainvoke(...)`` so that timing,
    provider attribution, and error translation happen in exactly one place.
    """
    started = time.perf_counter()
    try:
        response = await get_llm(temperature).ainvoke(messages)
    except Exception as exc:
        # Reaching here means the whole chain failed — the fallback's exception is
        # the one that surfaces. The primary's was already logged by FallbackWatcher.
        raise ProviderError(
            provider=_provider_for(get_settings().fallback_model),
            message=f"all providers failed: {type(exc).__name__}: {exc}",
            node=node,
        ) from exc
    _record(metrics, node, response, time.perf_counter() - started)
    return response


async def call_structured[SchemaT: BaseModel](
    node: str,
    messages: LanguageModelInput,
    schema: type[SchemaT],
    metrics: RunMetrics,
    temperature: float | None = None,
) -> SchemaT:
    """Invoke the composed model constrained to ``schema`` and return that type.

    ``with_structured_output`` declares ``dict | BaseModel``, so the result is
    narrowed with ``model_validate``. That gives mypy a concrete type without a
    ``type: ignore``, and re-validates the model's output as a side benefit.
    """
    started = time.perf_counter()
    try:
        raw = await get_structured_llm(schema, temperature).ainvoke(messages)
    except Exception as exc:
        raise ProviderError(
            provider=_provider_for(get_settings().fallback_model),
            message=f"all providers failed: {type(exc).__name__}: {exc}",
            node=node,
        ) from exc
    _record(metrics, node, raw, time.perf_counter() - started)
    return schema.model_validate(raw if isinstance(raw, dict) else raw.model_dump())
