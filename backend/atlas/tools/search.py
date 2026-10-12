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

from atlas.config import get_settings
from atlas.errors import SearchError
from atlas.graph.state import Source

logger = logging.getLogger(__name__)


def build_search_tool(max_results: int, include_content: bool = False) -> TavilySearch:
    """Construct the Tavily tool with an explicitly injected credential.

    The key must be passed here rather than left to Tavily's ambient lookup:
    pydantic-settings reads the root ``.env`` into :class:`Settings` without
    exporting anything to ``os.environ``, so the fallback would find nothing.
    Setting ``os.environ`` from config instead would be a hidden global mutation
    and is deliberately avoided.

    ``tavily_api_key`` is the supported parameter — ``TavilySearch.__init__``
    detects it and builds the wrapper itself. Pass a **plain str**, never a
    ``SecretStr``: the wrapper's ``mode="before"`` validator runs
    ``get_from_dict_or_env``, which stringifies whatever it is given, and
    ``str(SecretStr("tvly-..."))`` is ``'**********'``. Wrapping the secret sends
    ten asterisks as the credential and every search returns 401 against a
    perfectly valid key. ``test_search`` asserts the key that arrives equals the
    configured one, because a truthiness check passes on the mask.

    ``include_content`` asks for each page's text as plain text, alongside the
    snippet. Off unless asked: the prompts read snippets, and page text is heavy.
    """
    settings = get_settings()
    return TavilySearch(
        max_results=max_results,
        tavily_api_key=settings.tavily_api_key,
        include_raw_content="text" if include_content else False,
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
        # Its "raw_content" is the page text, our ``content``.
        snippet = item.get("content")
        page = item.get("raw_content")
        sources.append(
            Source(
                url=url,
                title=title if isinstance(title, str) else url,
                snippet=snippet if isinstance(snippet, str) else "",
                content=page if isinstance(page, str) and page else None,
            )
        )
    return sources


async def search(
    query: str, max_results: int | None = None, include_content: bool = False
) -> list[Source]:
    """Run one web search and return normalized sources.

    Raises :class:`SearchError` on any failure. Returning an empty list is a
    valid, non-exceptional outcome — "nothing found" is information, and the
    researcher decides whether to re-query.
    """
    settings = get_settings()
    limit = max_results if max_results is not None else settings.search_results_per_query

    tool = build_search_tool(limit, include_content)
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
