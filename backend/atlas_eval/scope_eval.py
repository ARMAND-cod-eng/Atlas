"""Measuring the scope check: telling a question for the AI Act and the GDPR from one that is not.

Four sets of questions, each labelled before anything was measured on it:

* **in scope**: the 40 compliance questions (``dataset.compliance.v1.jsonl``),
  and 24 casual ones written for this check (``scope.v1.jsonl``): short, plain,
  the way a person types ("right to be forgotten").
* **off-topic**: the 50 general research questions (``dataset.v1.jsonl``),
  about software and AI, written long before this check, and 30 everyday ones
  written for it, among them two EU laws the library does not hold (the DSA,
  NIS2).

Three ways of deciding are measured, as a run acts on each:

* **the guard** (``names_the_law``): free. It must never fire on an off-topic
  question, since it overrides the planner.
* **offline mode's stand-in** (``OFFLINE_FLOOR``): free. Its floor was set
  looking at every set here, so its numbers are in-sample.
* **the planner's judgment**, with the guard applied: one planner call per
  question, which costs. Its prompt was written before any of it was measured;
  a change to it is tried on the dev half (odd ids) and measured once on test.

Counted both ways: a question about the law turned away (a false refusal, the
worse failure) and an off-topic question caught.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from atlas.domains import using_domain
from atlas.graph.nodes import _plan
from atlas.library.scope import OFFLINE_FLOOR, closest_match, declined_reason, names_the_law
from atlas.metrics import RunMetrics
from atlas_eval.dataset import DATA_DIR, DATASETS, load_dataset

SCOPE_CASES_PATH = DATA_DIR / "scope.v1.jsonl"

Source = Literal["compliance", "casual", "general", "everyday"]
Split = Literal["dev", "test"]
Method = Literal["guard", "offline", "planner"]


class ScopeCase(BaseModel):
    """One labelled question."""

    id: str
    query: str
    in_scope: bool
    source: Source

    @property
    def split(self) -> Split:
        """Odd ids develop, even ids test, as in the retrieval eval."""
        return "dev" if int(re.sub(r"\D", "", self.id)) % 2 == 1 else "test"


class Decision(BaseModel):
    """What one method decided for one question."""

    id: str
    method: Method
    declined: bool
    reason: str | None = None


def load_scope_cases(path: Path = SCOPE_CASES_PATH) -> list[ScopeCase]:
    """Every labelled question: the two eval datasets and the ones written for this check."""
    cases = [
        *(
            ScopeCase(id=item.id, query=item.query, in_scope=True, source="compliance")
            for item in load_dataset(*DATASETS["compliance"]).items
        ),
        *(
            ScopeCase(id=item.id, query=item.query, in_scope=False, source="general")
            for item in load_dataset(*DATASETS["general"]).items
        ),
    ]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cases.append(
                ScopeCase(
                    id=row["id"],
                    query=row["query"],
                    in_scope=row["in_scope"],
                    source="casual" if row["in_scope"] else "everyday",
                )
            )
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate scope case ids")
    return cases


def by_guard(case: ScopeCase) -> Decision:
    """The guard alone: a question it holds for is planned. Here it never declines anything."""
    return Decision(id=case.id, method="guard", declined=not names_the_law(case.query))


def by_offline_stand_in(case: ScopeCase) -> Decision:
    declined = not names_the_law(case.query) and closest_match(case.query) < OFFLINE_FLOOR
    return Decision(id=case.id, method="offline", declined=declined)


Planner = Callable[[str], Awaitable[str | None]]


def compliance_planner(metrics: RunMetrics) -> Planner:
    """The production planner, in the compliance domain: its ``out_of_scope``, or None."""

    async def judge(query: str) -> str | None:
        with using_domain("compliance"):
            plan = await _plan(query, [], metrics)
        return plan.out_of_scope

    return judge


async def by_planner(
    cases: Sequence[ScopeCase], planner: Planner, concurrency: int = 5
) -> list[Decision]:
    """The planner's judgment, with the guard applied as a run applies it. Calls the model."""
    gate = asyncio.Semaphore(concurrency)

    async def one(case: ScopeCase) -> Decision:
        async with gate:
            judged = await planner(case.query)
        reason = declined_reason(judged, case.query)
        return Decision(id=case.id, method="planner", declined=reason is not None, reason=reason)

    return list(await asyncio.gather(*(one(case) for case in cases)))


class Tally(BaseModel):
    """One method on one source and split: how many questions, how many declined."""

    method: Method
    source: Source
    split: Split
    in_scope: bool
    questions: int
    declined: int


def tally(cases: Sequence[ScopeCase], decisions: Sequence[Decision]) -> list[Tally]:
    """Counted per method, source and split, in a fixed order."""
    case_of = {case.id: case for case in cases}
    counts: dict[tuple[Method, Source, Split], list[int]] = {}
    for decision in decisions:
        case = case_of[decision.id]
        key = (decision.method, case.source, case.split)
        seen = counts.setdefault(key, [0, 0])
        seen[0] += 1
        seen[1] += int(decision.declined)
    sources: tuple[Source, ...] = ("compliance", "casual", "general", "everyday")
    methods: tuple[Method, ...] = ("guard", "offline", "planner")
    splits: tuple[Split, ...] = ("dev", "test")
    return [
        Tally(
            method=method,
            source=source,
            split=split,
            in_scope=source in ("compliance", "casual"),
            questions=counts[(method, source, split)][0],
            declined=counts[(method, source, split)][1],
        )
        for method in methods
        for source in sources
        for split in splits
        if (method, source, split) in counts
    ]
