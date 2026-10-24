"""The regulation library over HTTP: a provision's text, by its citation label."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atlas.library import ACTS, load_library
from atlas.library.citations import cited_labels
from atlas.library.sources import as_source

router = APIRouter()


class Provision(BaseModel):
    """One provision, as a citation chip shows it."""

    label: str
    # The act's short name: "AI Act", "GDPR".
    act: str
    # The heading of the article or annex it belongs to.
    title: str
    text: str
    # The Official Journal, at the article, annex or recital.
    url: str


@router.get("/library/provision", response_model=Provision)
async def provision(label: str) -> Provision:
    """The provision a citation names, read as the citation check reads it.

    "GDPR Article 33(1)" and "GDPR Art 33(1)" are both "GDPR Art. 33(1)". A label
    that names nothing is 404, never a near match: a chip that showed Article 33
    for a citation of Article 34 would vouch for the wrong law.
    """
    library = load_library()
    canonical = cited_labels(f"[{label}]")
    unit = library.resolve(canonical[0]) if canonical else None
    if unit is None:
        raise HTTPException(status_code=404, detail=f"no such provision: {label}")
    return Provision(
        label=unit.label,
        act=ACTS[unit.act].short,
        title=unit.title,
        text=unit.text,
        url=as_source(library, unit).url,
    )
