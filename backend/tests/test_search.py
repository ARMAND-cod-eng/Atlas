"""The search tool: normalization in, typed errors out.

Tavily is mocked entirely — nothing here touches the network.
"""

from __future__ import annotations

from typing import Any

import pytest

from atlas.errors import SearchError
from atlas.tools import search as search_module
from tests.conftest import FAKE_ENV

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


# --- the credential actually arriving --------------------------------------


def test_configured_key_reaches_the_client() -> None:
    """The key the tool holds must EQUAL the configured value.

    Not a truthiness check. A truthiness check is exactly what passed while the
    tool was holding '**********': the previous implementation passed a SecretStr
    into TavilySearchAPIWrapper, whose mode="before" validator stringifies via
    get_from_dict_or_env, so str(SecretStr(...)) — the mask — became the
    credential. Ten asterisks were sent to Tavily and every search 401'd against
    a perfectly valid key.
    """
    tool = search_module.build_search_tool(5)
    held = tool.api_wrapper.tavily_api_key.get_secret_value()

    assert held == FAKE_ENV["TAVILY_API_KEY"]


def test_key_is_not_the_secretstr_mask() -> None:
    """Name the failure mode so a future re-wrap fails here, not in production."""
    tool = search_module.build_search_tool(5)
    held = tool.api_wrapper.tavily_api_key.get_secret_value()

    assert set(held) != {"*"}, "the SecretStr mask is being sent as the credential"


def test_max_results_reaches_the_client() -> None:
    assert search_module.build_search_tool(7).max_results == 7


# --- in-band errors -----------------------------------------------------------


async def test_in_band_error_payload_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tavily reports failures in the payload, not by raising.

    This is the exact observed shape: an Exception *object* under "error", and no
    "results" key at all. Read naively it is indistinguishable from an honest
    empty search — every researcher silently reports zero sources and the run
    completes looking healthy.
    """
    _patch_tool(monkeypatch, FakeTool(payload={"error": Exception("Error 401: Unauthorized")}))

    with pytest.raises(SearchError) as excinfo:
        await search_module.search("anything")

    assert "401" in str(excinfo.value)
    assert "Unauthorized" in str(excinfo.value)


async def test_error_string_also_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The error value is an Exception today; a plain string must work too."""
    _patch_tool(monkeypatch, FakeTool(payload={"error": "Error 432: Rate limited"}))

    with pytest.raises(SearchError):
        await search_module.search("anything")


async def test_error_alongside_results_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial failure must not be read as a success.

    If an error key can be ignored whenever some results came back, a degraded
    search silently becomes an authoritative one.
    """
    payload = {
        "error": Exception("Error 401: Unauthorized"),
        "results": [{"url": "https://partial.example", "title": "T", "content": "c"}],
    }
    _patch_tool(monkeypatch, FakeTool(payload=payload))

    with pytest.raises(SearchError):
        await search_module.search("partial")


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
