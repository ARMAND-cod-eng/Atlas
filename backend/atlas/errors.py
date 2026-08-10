"""Typed errors Atlas raises deliberately.

These exist so the API layer can turn a failure into a response that says what
broke and where, instead of a bare 500 with an empty body. Phase 0 established
the cost of not having this: every defect took ten diagnostic steps instead of
one, because the only signal was "Internal Server Error".
"""

from __future__ import annotations

from typing import Any


class AtlasError(Exception):
    """Base for every error Atlas raises on purpose.

    Anything inheriting from this is a known failure mode with a useful message.
    Anything that does not is a bug, and is reported as such.
    """


class ProviderError(AtlasError):
    """An LLM call failed after the whole fallback chain was exhausted.

    Carries the provider so the response can name it. ``node`` is filled in by
    whichever graph node was running, which is usually the first thing you want
    to know.
    """

    def __init__(self, provider: str, message: str, *, node: str | None = None) -> None:
        self.provider = provider
        self.node = node
        super().__init__(message)

    def __reduce__(self) -> tuple[Any, ...]:
        """Survive a trip through pickle.

        Needed since Phase 3: a run executes in the worker, and arq pickles a
        failing job's exception so the awaiting request can re-raise it. The
        default reconstructor calls ``cls(*self.args)`` — one positional — which
        does not match this signature, so the error would arrive as a confusing
        ``TypeError`` about a missing ``message`` and the real provider failure
        would be lost on the way to the client.
        """
        return (self.__class__, (self.provider, str(self)), {"node": self.node})


class SearchError(AtlasError):
    """A web search failed.

    Raised instead of letting a raw HTTP error escape the tool layer, so callers
    can catch one predictable type rather than whatever the vendor SDK throws.
    """

    def __init__(self, message: str, *, query: str | None = None) -> None:
        self.query = query
        super().__init__(message)

    def __reduce__(self) -> tuple[Any, ...]:
        """Survive a trip through pickle — see :meth:`ProviderError.__reduce__`."""
        return (self.__class__, (str(self),), {"query": self.query})
