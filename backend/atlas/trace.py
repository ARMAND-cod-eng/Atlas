"""What each agent was told, and what it said back, as it actually happened.

Recorded at the two places every call passes through: ``atlas.models`` for the
model, the researcher's search helpers for the tools. Never reconstructed from
state and today's templates, which would show what a prompt *would* have been
and differ silently the day a template changes.

A call is attributed to the span that made it. ``events.node_span`` opens a span
per node execution and sets it here; concurrent researchers are separate asyncio
tasks, so each sees its own. A call made outside any span (the eval's
single-call baseline) is recorded with no span, and nowhere when no graph is
streaming: ``emit`` is a no-op then.

The record travels as an ``agent_call`` custom event. The runner stores it whole
and puts only :meth:`AgentCall.summary` on the live stream, which replay sends
to every reconnecting client.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Characters of a call's output shown on the live stream.
PREVIEW_CHARS = 280


@dataclass(frozen=True, slots=True)
class Span:
    """One execution of one node: the planner once, a researcher per attempt."""

    id: str
    node: str
    subtask_id: str | None


_SPAN: ContextVar[Span | None] = ContextVar("atlas_span", default=None)


@contextmanager
def span(node: str, subtask_id: str | None) -> Iterator[Span]:
    current = Span(id=uuid4().hex[:12], node=node, subtask_id=subtask_id)
    token = _SPAN.set(current)
    try:
        yield current
    finally:
        _SPAN.reset(token)


def current_span() -> Span | None:
    return _SPAN.get()


class AgentCall(BaseModel):
    """One call an agent made: to a model, or to a search."""

    call_id: str = Field(default_factory=lambda: uuid4().hex)
    span_id: str | None = None
    node: str
    subtask_id: str | None = None
    kind: Literal["model", "tool"]
    # The model that answered, or the tool: "library", "web".
    name: str
    started_at: datetime
    duration_s: float
    # A model's messages as [{"role", "content"}]; a tool's {"query": ...}.
    request: Any
    # A model's text or parsed output; a tool's results.
    response: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        """What the live stream carries: enough to list the call, not its contents."""
        text = self._readable()
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "..."
        return {
            "call_id": self.call_id,
            "span_id": self.span_id,
            "subtask_id": self.subtask_id,
            "kind": self.kind,
            "name": self.name,
            "duration_s": self.duration_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "preview": preview,
        }

    def _readable(self) -> str:
        """The response as a person skims it: a search's hits, an output as JSON."""
        if self.response is None or isinstance(self.response, str):
            return self.response or ""
        if self.kind == "tool" and isinstance(self.response, list):
            return "; ".join(
                str(ref.get("label") or ref.get("url"))
                for ref in self.response
                if isinstance(ref, Mapping)
            )
        return json.dumps(self.response, default=str)


def as_messages(messages: object) -> list[dict[str, str]]:
    """A model input as role and text, the shape the inspector shows."""
    if isinstance(messages, str):
        return [{"role": "human", "content": messages}]
    if isinstance(messages, Sequence):
        return [
            {"role": message.type, "content": message.text}
            if isinstance(message, BaseMessage)
            else {"role": "unknown", "content": str(message)}
            for message in messages
        ]
    return [{"role": "unknown", "content": str(messages)}]


def record(
    *,
    kind: Literal["model", "tool"],
    name: str,
    node: str,
    started_at: datetime,
    duration_s: float,
    request: object,
    response: object = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error: str | None = None,
) -> None:
    """Emit one call, attributed to the current span. Never raises."""
    # Imported here: atlas.events imports this module for its spans.
    from atlas.events import AGENT_CALL, emit

    current = current_span()
    call = AgentCall(
        span_id=current.id if current else None,
        node=node,
        subtask_id=current.subtask_id if current else None,
        kind=kind,
        name=name,
        started_at=started_at,
        duration_s=duration_s,
        request=request,
        response=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
    )
    # The node travels as the event's own, not twice.
    emit(AGENT_CALL, node, **call.model_dump(mode="json", exclude={"node"}))


def now() -> datetime:
    return datetime.now(UTC)


def result_refs(sources: Sequence[Mapping[str, Any] | BaseModel]) -> list[dict[str, Any]]:
    """What a search returned, as the inspector lists it: label or URL, and title."""
    refs: list[dict[str, Any]] = []
    for source in sources:
        data = source.model_dump() if isinstance(source, BaseModel) else dict(source)
        ref = {"url": data.get("url"), "title": data.get("title")}
        if data.get("label"):
            ref = {"label": data["label"], "title": data.get("title")}
        refs.append(ref)
    return refs
