"""Provisions of the law, as sources a researcher reads and a citation names."""

from __future__ import annotations

import functools
from urllib.parse import urlparse

from atlas.graph.state import Source
from atlas.library import ACTS, Library, Unit, load_library
from atlas.library.search import LibrarySearch

# A provision's text as a researcher reads it. Most are a few hundred characters;
# a whole annex looked up by reference can run to seventeen thousand, and the
# point a question needs is then better found by its own label.
LAW_TEXT_CHARS = 4_000

# Where the EU publishes guidance on these two acts: the Commission (and its AI
# Office), the EDPB, the EDPS. Their pages interpret the law with authority a
# blog does not have, though still not the law's own.
_OFFICIAL_HOSTS = ("europa.eu",)


@functools.lru_cache(maxsize=1)
def library_search() -> LibrarySearch:
    """One index per process: building it parses both acts."""
    return LibrarySearch(load_library())


def _anchor(library: Library, unit: Unit) -> str:
    """The Official Journal anchor of the article, annex or recital a unit is in."""
    top = unit
    while top.parent is not None:
        top = library.units[top.parent]
    kind, number = top.id.split("/")[1:3]
    return {"art": "art_", "anx": "anx_", "rct": "rct_"}[kind] + number


def as_source(library: Library, unit: Unit) -> Source:
    text = unit.text
    if len(text) > LAW_TEXT_CHARS:
        text = text[:LAW_TEXT_CHARS].rsplit(" ", 1)[0] + " [...]"
    return Source(
        url=f"{ACTS[unit.act].eli}#{_anchor(library, unit)}",
        title=unit.title or unit.label,
        snippet=text,
        authority="primary_law",
        label=unit.label,
    )


def law_sources(question: str, limit: int) -> list[Source]:
    """The provisions the library search returns for a question, best first."""
    search = library_search()
    return [as_source(search.library, hit.unit) for hit in search.search(question, limit)]


def with_authority(source: Source) -> Source:
    """A web source, marked official guidance when an EU institution published it."""
    host = urlparse(source.url).hostname or ""
    official = any(host == h or host.endswith(f".{h}") for h in _OFFICIAL_HOSTS)
    return source.model_copy(update={"authority": "official_guidance" if official else "secondary"})
