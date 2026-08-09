"""The search tool: normalization in, typed errors out.

Tavily is mocked entirely — nothing here touches the network.
"""

from __future__ import annotations

from typing import Any

import pytest

from atlas.errors import SearchError
from atlas.tools import search as search_module

TAVILY_PAYLOAD: dict[str, Any] = {
    "query": "what is RAG",
    "results": [
        {
            "title": "Retrieval-Augmented Generation",
            "url": "https://a.example/rag",
            "content": "RAG combines retrieval with generation.",
            "score": 0.98,
        },
        {
            "title": "Vector databases",
            "url": "https://b.example/vectors",
            "content": "Vectors index a corpus for similarity search.",
            "score": 0.91,
        },
    ],
}


class FakeTool:
    """Stands in for TavilySearch."""

    def __init__(self, payload: object = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, arg: dict[str, Any]) -> object:
        self.calls.append(arg)
        if self.error is not None:
            raise self.error
        return self.payload


def _patch_tool(monkeypatch: pytest.MonkeyPatch, tool: FakeTool) -> None:
    monkeypatch.setattr(search_module, "build_search_tool", lambda max_results: tool)


async def test_normalizes_tavily_shape_to_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tavily's field names stop at this module; nodes only ever see Source."""
    _patch_tool(monkeypatch, FakeTool(payload=TAVILY_PAYLOAD))

    sources = await search_module.search("what is RAG")

    assert len(sources) == 2
    assert sources[0].url == "https://a.example/rag"
    assert sources[0].title == "Retrieval-Augmented Generation"
    # Tavily calls the extract "content"; we call it a snippet everywhere.
    assert sources[0].snippet == "RAG combines retrieval with generation."


async def test_transport_failure_raises_search_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers catch one predictable type, not whatever the vendor SDK throws."""
    _patch_tool(monkeypatch, FakeTool(error=ConnectionError("tavily unreachable")))

    with pytest.raises(SearchError) as excinfo:
        await search_module.search("anything")

    assert excinfo.value.query == "anything"
    assert "ConnectionError" in str(excinfo.value)


async def test_in_band_error_payload_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tavily reports failures in the payload, not by raising.

    A 401 arrives as {"error": Exception("Error 401: Unauthorized")} with no
    "results" key. Read naively that is indistinguishable from an honest empty
    search — every researcher silently reports zero sources and the run completes
    looking healthy. This is the check that tells them apart.
    """
    _patch_tool(monkeypatch, FakeTool(payload={"error": Exception("Error 401: Unauthorized")}))

    with pytest.raises(SearchError) as excinfo:
        await search_module.search("anything")

    assert "401" in str(excinfo.value)


async def test_empty_results_are_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Nothing found" is information; the researcher decides whether to re-query."""
    _patch_tool(monkeypatch, FakeTool(payload={"results": []}))

    assert await search_module.search("obscure") == []


async def test_result_count_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"results": [{"url": f"https://e.example/{i}"} for i in range(20)]}
    _patch_tool(monkeypatch, FakeTool(payload=payload))

    sources = await search_module.search("many", max_results=3)

    assert len(sources) == 3


async def test_results_without_a_url_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result that cannot be cited is useless to a research system."""
    payload = {
        "results": [
            {"title": "no url here", "content": "..."},
            {"url": "https://good.example", "title": "Good", "content": "c"},
        ]
    }
    _patch_tool(monkeypatch, FakeTool(payload=payload))

    sources = await search_module.search("mixed")

    assert [s.url for s in sources] == ["https://good.example"]


async def test_missing_fields_fall_back_rather_than_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degrade to a usable Source instead of losing the whole search."""
    payload = {"results": [{"url": "https://bare.example"}]}
    _patch_tool(monkeypatch, FakeTool(payload=payload))

    sources = await search_module.search("bare")

    assert sources[0].title == "https://bare.example"
    assert sources[0].snippet == ""


async def test_malformed_payload_yields_no_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shape change upstream should degrade, not explode."""
    _patch_tool(monkeypatch, FakeTool(payload=["not", "a", "mapping"]))

    assert await search_module.search("weird") == []


async def test_query_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = FakeTool(payload=TAVILY_PAYLOAD)
    _patch_tool(monkeypatch, tool)

    await search_module.search("specific phrasing")

    assert tool.calls == [{"query": "specific phrasing"}]
