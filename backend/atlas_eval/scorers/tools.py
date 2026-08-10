"""Tool-call validity: deterministic wherever it can be.

Five of the six checks are mechanical — referential integrity, decomposition
width, redundant searches, argument bounds, retry bounds. Only one needs a judge:
whether the subtasks are genuinely *independent* and on-topic, which is the
property the planner prompt actually promises and the one thing here that resists
mechanisation.

The realised deterministic:judge split is computed per run and reported rather
than asserted, because a claim like "mostly deterministic" decays quietly as
checks get added.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atlas.config import get_settings
from atlas.graph.state import Subtask
from atlas_eval.dataset import EvalItem
from atlas_eval.scorers.completion import normalize
from atlas_eval.scorers.judge import judge

JUDGE_SYSTEM = """You are reviewing how a research system decomposed a question.

You are given the original QUESTION and the SUBTASKS it was split into. Judge two \
things about the set as a whole:

1. INDEPENDENCE — can each subtask be researched without knowing another's answer? \
A subtask that says "based on the above" or that only makes sense after another has \
been answered breaks this.
2. RELEVANCE — does each subtask serve the original question? A subtask that wanders \
off the topic breaks this.

Do not judge the number of subtasks; that is checked separately. Do not judge the \
wording. Answer `independent` and `relevant` as booleans, with one short reason each.
"""


class PlanVerdict(BaseModel):
    independent: bool
    relevant: bool
    reason: str = ""


class Check(BaseModel):
    """One validity check and how it came out."""

    name: str
    kind: str  # "deterministic" | "judge"
    passed: bool
    detail: str = ""


class ToolScore(BaseModel):
    score: float
    checks: list[Check] = Field(default_factory=list)

    @property
    def deterministic_count(self) -> int:
        return sum(1 for c in self.checks if c.kind == "deterministic")

    @property
    def judge_count(self) -> int:
        return sum(1 for c in self.checks if c.kind == "judge")


def _near_duplicate(a: str, b: str) -> bool:
    """Same search twice, allowing for rewording.

    Exact equality would miss the actual waste: a researcher retrying with
    "X overview" after "X" is the retry ladder working as designed, but two
    branches independently issuing near-identical searches is redundant work.
    Jaccard over tokens at 0.9 separates those.
    """
    left, right = set(normalize(a).split()), set(normalize(b).split())
    if not left or not right:
        return False
    overlap = len(left & right) / len(left | right)
    return overlap >= 0.9


async def score_tools(
    item: EvalItem,
    subtasks: list[Subtask],
    result_subtask_ids: list[str],
    search_calls: list[str],
    researcher_run_counts: dict[str, int],
) -> ToolScore:
    """Fraction of validity checks passed."""
    settings = get_settings()
    checks: list[Check] = []
    subtask_ids = [subtask.id for subtask in subtasks]

    # 1. Referential integrity: every result belongs to a planned subtask.
    unknown = sorted(set(result_subtask_ids) - set(subtask_ids))
    checks.append(
        Check(
            name="result_subtask_ids_known",
            kind="deterministic",
            passed=not unknown,
            detail=f"unknown ids: {unknown}" if unknown else "all results map to a subtask",
        )
    )

    # 2. Decomposition width against the hand-labelled range. This is the check
    #    that catches a planner emitting five subtasks for a single-fact question.
    count = len(subtask_ids)
    within = item.expected_subtasks.min <= count <= item.expected_subtasks.max
    checks.append(
        Check(
            name="subtask_count_reasonable",
            kind="deterministic",
            passed=within,
            detail=f"{count} subtasks, expected "
            f"{item.expected_subtasks.min}-{item.expected_subtasks.max}",
        )
    )

    # 3. No redundant duplicate searches.
    duplicates = [
        (a, b)
        for i, a in enumerate(search_calls)
        for b in search_calls[i + 1 :]
        if _near_duplicate(a, b)
    ]
    checks.append(
        Check(
            name="no_duplicate_searches",
            kind="deterministic",
            passed=not duplicates,
            detail=f"{len(duplicates)} near-duplicate pair(s)"
            if duplicates
            else f"{len(search_calls)} distinct searches",
        )
    )

    # 4. Search count bounded by the configured retry ladder. More searches than
    #    branches x max attempts means the ladder is not being respected.
    ceiling = (
        max(1, count)
        * settings.max_tool_calls_per_researcher
        * (settings.max_critic_iterations + 1)
    )
    checks.append(
        Check(
            name="search_count_within_bounds",
            kind="deterministic",
            passed=len(search_calls) <= ceiling,
            detail=f"{len(search_calls)} searches, ceiling {ceiling}",
        )
    )

    # 5. No branch re-ran more than the critic loop allows.
    cap = settings.max_critic_iterations + 1
    over = {sid: n for sid, n in researcher_run_counts.items() if n > cap}
    checks.append(
        Check(
            name="retry_count_within_cap",
            kind="deterministic",
            passed=not over,
            detail=f"over cap: {over}"
            if over
            else f"max runs {max(researcher_run_counts.values(), default=0)}, cap {cap}",
        )
    )

    # 6. The one that needs a judge. It gets the subtask *questions* — judging
    #    independence from opaque ids would be scoring nothing at all.
    if subtasks:
        verdict = await judge(
            PlanVerdict,
            JUDGE_SYSTEM,
            f"QUESTION:\n{item.query}\n\nSUBTASKS:\n"
            + "\n".join(f"- {subtask.question}" for subtask in subtasks),
        )
        checks.append(
            Check(
                name="subtasks_independent_and_relevant",
                kind="judge",
                passed=verdict.independent and verdict.relevant,
                detail=verdict.reason,
            )
        )
    else:
        checks.append(
            Check(
                name="subtasks_independent_and_relevant",
                kind="judge",
                passed=False,
                detail="no subtasks were produced",
            )
        )

    passed = sum(1 for c in checks if c.passed)
    return ToolScore(score=passed / len(checks) if checks else 0.0, checks=checks)
