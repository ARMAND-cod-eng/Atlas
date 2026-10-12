"""Scoring a run record across all three categories.

Kept separate from the runner so scoring can be re-done — with a revised judge
prompt, say — without re-running the agent. The run artefact is the expensive
thing; scoring it again is cheap.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from pydantic import BaseModel, Field

from atlas.graph.nodes import RESEARCHER
from atlas_eval.dataset import EvalItem
from atlas_eval.runner import RunRecord
from atlas_eval.scorers.completion import CompletionScore, score_completion
from atlas_eval.scorers.grounding import GroundingScore, score_grounding
from atlas_eval.scorers.tools import ToolScore, score_tools


class ItemScore(BaseModel):
    item_id: str
    completion: CompletionScore
    grounding: GroundingScore
    tools: ToolScore
    error: str | None = None

    @property
    def as_dict(self) -> dict[str, float]:
        """Every category that applies to this run."""
        scores = {"completion": self.completion.score, "grounding": self.grounding.score}
        if not self.tools.not_applicable:
            scores["tools"] = self.tools.score
        return scores


class SuiteScore(BaseModel):
    scores: list[ItemScore] = Field(default_factory=list)

    def means(self) -> dict[str, float]:
        """Per-category means over the runs each category applies to."""
        return {
            category: sum(scores.values()) / len(scores)
            for category, scores in self.per_item().items()
        }

    def per_item(self) -> dict[str, dict[str, float]]:
        """``{category: {item_id: score}}`` — what the paired bootstrap needs.

        A category that applies to no run is absent rather than empty, so a
        baseline with no tool use cannot be gated as if it had scored nothing.
        """
        per_item: dict[str, dict[str, float]] = {}
        for category in ("completion", "grounding", "tools"):
            scores = {s.item_id: s.as_dict[category] for s in self.scores if category in s.as_dict}
            if scores:
                per_item[category] = scores
        return per_item

    def check_split(self) -> dict[str, tuple[int, int]]:
        """Realised deterministic:judge counts, per category.

        Computed rather than claimed. "Mostly deterministic" is exactly the sort
        of statement that stays true for a week after it is written.
        """
        completion_det = sum(s.completion.deterministic_count for s in self.scores)
        completion_judge = sum(s.completion.judge_count for s in self.scores)
        tools_det = sum(s.tools.deterministic_count for s in self.scores)
        tools_judge = sum(s.tools.judge_count for s in self.scores)
        grounding_claims = sum(s.grounding.claims_total for s in self.scores)
        return {
            "completion": (completion_det, completion_judge),
            "tools": (tools_det, tools_judge),
            # Grounding is entirely judged; the count is claims verified.
            "grounding": (0, grounding_claims),
        }


def researcher_runs(record: RunRecord) -> dict[str, int]:
    """How many times each subtask's researcher actually executed.

    Counted from node timings, one per execution — not from ``research_results``,
    which the upsert reducer keeps at one entry per subtask however often it was
    retried. Counting results gave every subtask exactly one run, so the
    retry-cap check that reads this could not fail.
    """
    return dict(
        Counter(
            timing.subtask_id
            for timing in record.metrics.node_timings
            if timing.node == RESEARCHER and timing.subtask_id is not None
        )
    )


def _crashed(item: EvalItem, record: RunRecord) -> ItemScore:
    """Zero in every category, with nothing sent to the judge.

    Scored normally, a crash did well: its empty answer decomposed into no
    claims, which grounding reads as an honest abstention (1.0), and four of the
    six tool checks passed vacuously on an empty plan. A regression that crashed
    more items pushed the grounding mean *up*.
    """
    return ItemScore(
        item_id=item.id,
        completion=CompletionScore(score=0.0),
        grounding=GroundingScore(score=0.0),
        tools=ToolScore(score=0.0, not_applicable=record.variant == "single-call"),
        error=record.error,
    )


async def score_run(item: EvalItem, record: RunRecord) -> ItemScore:
    """Score one run across all three categories."""
    if record.error is not None:
        return _crashed(item, record)

    if record.variant == "single-call":
        # No plan, no researchers: tool use does not apply, and scoring it would
        # charge the baseline for checks it cannot pass by design.
        completion, grounding = await asyncio.gather(
            score_completion(item, record.answer),
            score_grounding(record.answer, record.retrieved_context),
        )
        tools = ToolScore(score=0.0, not_applicable=True)
    else:
        completion, grounding, tools = await asyncio.gather(
            score_completion(item, record.answer),
            score_grounding(record.answer, record.retrieved_context),
            score_tools(
                item,
                record.subtasks,
                [r.subtask_id for r in record.research_results],
                record.search_calls,
                researcher_runs(record),
            ),
        )
    return ItemScore(
        item_id=item.id,
        completion=completion,
        grounding=grounding,
        tools=tools,
        error=record.error,
    )


async def score_all(pairs: list[tuple[EvalItem, RunRecord]], concurrency: int = 4) -> SuiteScore:
    limiter = asyncio.Semaphore(concurrency)

    async def guarded(item: EvalItem, record: RunRecord) -> ItemScore:
        async with limiter:
            return await score_run(item, record)

    scores = await asyncio.gather(*(guarded(i, r) for i, r in pairs))
    return SuiteScore(scores=list(scores))
