"""Scoring a run record across all three categories.

Kept separate from the runner so scoring can be re-done — with a revised judge
prompt, say — without re-running the agent. The run artefact is the expensive
thing; scoring it again is cheap.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from pydantic import BaseModel, Field

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
        return {
            "completion": self.completion.score,
            "grounding": self.grounding.score,
            "tools": self.tools.score,
        }


class SuiteScore(BaseModel):
    scores: list[ItemScore] = Field(default_factory=list)

    def means(self) -> dict[str, float]:
        if not self.scores:
            return {}
        return {
            category: sum(s.as_dict[category] for s in self.scores) / len(self.scores)
            for category in ("completion", "grounding", "tools")
        }

    def per_item(self) -> dict[str, dict[str, float]]:
        """``{category: {item_id: score}}`` — what the paired bootstrap needs."""
        return {
            category: {s.item_id: s.as_dict[category] for s in self.scores}
            for category in ("completion", "grounding", "tools")
        }

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


async def score_run(item: EvalItem, record: RunRecord) -> ItemScore:
    """Score one run across all three categories."""
    run_counts = Counter(result.subtask_id for result in record.research_results)

    completion, grounding, tools = await asyncio.gather(
        score_completion(item, record.answer),
        score_grounding(record.answer, record.retrieved_context),
        score_tools(
            item,
            record.subtasks,
            [r.subtask_id for r in record.research_results],
            record.search_calls,
            dict(run_counts),
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
