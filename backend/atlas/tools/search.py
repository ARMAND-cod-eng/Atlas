"""Web search via Tavily, normalized to Atlas's own :class:`Source`.

Two jobs, both about containment:

* Tavily's response shape stops here. Nodes see ``list[Source]``, so replacing
  the search provider is a change to this file and nothing else.
* Transport and API failures become :class:`SearchError`. Callers catch one
  predictable type instead of whatever the vendor SDK happens to raise.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_tavily import TavilySearch
from langchain_tavily.tavily_search import TavilySearchAPIWrapper
from pydantic import SecretStr

from atlas.config import get_settings
from atlas.errors import SearchError
from atlas.graph.state import Source

logger = logging.getLogger(__name__)


def build_search_tool(max_results: int) -> TavilySearch:
    """Construct the Tavily tool with an explicitly injected credential.

    The key is passed through an api_wrapper rather than left to Tavily's
    environment lookup: pydantic-settings reads the root ``.env`` into
    :class:`Settings` without exporting anything to ``os.environ``, so the
    ambient lookup would find nothing and fail at call time with a confusing
    auth error rather than at startup.
    """
    settings = get_settings()
    return TavilySearch(
        max_results=max_results,
        api_wrapper=TavilySearchAPIWrapper(tavily_api_key=SecretStr(settings.tavily_api_key)),
    )


def _raise_if_error(payload: object, query: str) -> None:
    """Turn Tavily's in-band error into a real exception.

    TavilySearch catches its own transport and API failures and returns
    ``{"error": <Exception>}`` instead of raising, so the try/except around
    ``ainvoke`` never fires. Without this check a 401 arrives as a payload with no
    ``results`` key and is indistinguishable from an honest "nothing found" —
    every researcher reports zero sources, the run completes, and the only clue is
    an answer that quietly admits it had nothing to cite.
    """
    if not isinstance(payload, Mapping):
        return
    error = payload.get("error")
    if error:
        raise SearchError(f"search failed: {error}", query=query)


def _coerce_sources(payload: object, limit: int) -> list[Source]:
    """Pull ``Source`` records out of whatever Tavily returned.

    Defensive on purpose. The tool is typed as returning ``dict[str, Any]``, but
    a malformed or partial result should degrade to fewer sources rather than
    raise — a researcher with two usable sources is far more useful than one that
    died on a missing field.
    """
    if not isinstance(payload, Mapping):
        return []
    raw_results = payload.get("results")
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, str | bytes):
        return []

    sources: list[Source] = []
    for item in raw_results[:limit]:
        if not isinstance(item, Mapping):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            # A result with no URL cannot be cited, which makes it useless here.
            continue
        title = item.get("title")
        # Tavily calls the extract "content"; we call it a snippet everywhere else.
        snippet = item.get("content")
        sources.append(
            Source(
                url=url,
                title=title if isinstance(title, str) else url,
                snippet=snippet if isinstance(snippet, str) else "",
            )
        )
    return sources


async def search(query: str, max_results: int | None = None) -> list[Source]:
    """Run one web search and return normalized sources.

    Raises :class:`SearchError` on any failure. Returning an empty list is a
    valid, non-exceptional outcome — "nothing found" is information, and the
    researcher decides whether to re-query.
    """
    settings = get_settings()
    limit = max_results if max_results is not None else settings.search_results_per_query

    tool = build_search_tool(limit)
    try:
        raw: Any = await tool.ainvoke({"query": query})
    except Exception as exc:
        raise SearchError(
            f"search failed: {type(exc).__name__}: {exc}",
            query=query,
        ) from exc

    # Must come before normalization: an error payload has no "results" key and
    # would otherwise normalize to an empty list, hiding the failure.
    _raise_if_error(raw, query)

    sources = _coerce_sources(raw, limit)
    logger.info("search complete", extra={"result_count": len(sources)})
    return sources
