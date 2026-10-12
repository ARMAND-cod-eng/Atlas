"""Scoring a run, and what a broken run is allowed to score.

Three defects, one shape: the eval reported something better than what happened.

* A run that **crashed** scored 1.0 on grounding and 0.67 on tools. Its empty
  answer decomposed into no claims, which the grounding scorer reads as an honest
  abstention, and four of the six tool checks pass vacuously on an empty plan. A
  regression that crashed more items pushed the grounding mean *up*.
* ``retry_count_within_cap`` read its counts from ``research_results``, which the
  upsert reducer keeps at one entry per subtask. It could not fail.
* ``atlas_eval run`` exited 0 however many items crashed, so the CI smoke — whose
  whole job is to catch crashes — passed with every item dead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from atlas.graph.state import ResearchResult, Subtask
from atlas.metrics import NodeTiming, RunMetrics
from atlas_eval import cli
from atlas_eval.dataset import load_dataset
from atlas_eval.runner import RunRecord
from atlas_eval.score import researcher_runs, score_run
from atlas_eval.scorers import completion, grounding, tools
from atlas_eval.scorers.completion import FactVerdict
from atlas_eval.scorers.grounding import ClaimList, VerdictList, score_grounding
from atlas_eval.scorers.tools import PlanVerdict, score_tools


@pytest.fixture
def judge_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A judge that approves everything, and records which schemas it was asked for."""
    calls: list[str] = []

    async def approving_judge(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        calls.append(schema.__name__)
        if schema is FactVerdict:
            return FactVerdict(present=True, evidence="e")
        if schema is ClaimList:
            return ClaimList(claims=[])
        if schema is VerdictList:
            return VerdictList(verdicts=[])
        if schema is PlanVerdict:
            return PlanVerdict(independent=True, relevant=True)
        raise AssertionError(f"unexpected judge schema {schema.__name__}")

    for module in (completion, grounding, tools):
        monkeypatch.setattr(module, "judge", approving_judge)
    return calls


def _item_and_plan(subtask_count: int = 2) -> tuple[Any, list[Subtask]]:
    item = load_dataset().items[0]
    plan = [Subtask(id=f"s{i}", question=f"q{i}", rationale="r") for i in range(subtask_count)]
    return item, plan


# --- a crashed run --------------------------------------------------------------


async def test_a_crashed_run_scores_zero_in_every_category(judge_calls: list[str]) -> None:
    item, _ = _item_and_plan()
    crashed = RunRecord(item_id=item.id, query=item.query, answer="", error="ProviderError: x")

    score = await score_run(item, crashed)

    assert score.as_dict == {"completion": 0.0, "grounding": 0.0, "tools": 0.0}
    assert score.error == "ProviderError: x"
    # Nothing to judge, so nothing is spent judging it.
    assert judge_calls == []


async def test_a_blank_answer_is_not_an_abstention(judge_calls: list[str]) -> None:
    """No answer at all is not the same as an answer that asserts nothing."""
    blank = await score_grounding("", "some retrieved context")

    assert (blank.score, blank.not_applicable) == (0.0, False)


async def test_an_answer_with_no_factual_claims_is_still_an_abstention(
    judge_calls: list[str],
) -> None:
    """The control: the abstention path the failure-expected items rely on survives."""
    honest = await score_grounding("The sources do not cover this.", "context")

    assert (honest.score, honest.not_applicable) == (1.0, True)
    assert judge_calls == ["ClaimList"]


# --- retries are counted from executions ------------------------------------------


def _record_with_researcher_runs(runs: dict[str, int]) -> RunRecord:
    item, plan = _item_and_plan(len(runs))
    timings = [
        NodeTiming(node="researcher", subtask_id=subtask_id, duration_s=1.0)
        for subtask_id, count in runs.items()
        for _ in range(count)
    ]
    return RunRecord(
        item_id=item.id,
        query=item.query,
        answer="an answer",
        subtasks=plan,
        # One entry per subtask, however often it ran — that is the reducer's job.
        research_results=[ResearchResult(subtask_id=sid, findings="f") for sid in runs],
        metrics=RunMetrics(node_timings=timings),
    )


def test_researcher_runs_counts_executions_not_surviving_results() -> None:
    record = _record_with_researcher_runs({"s0": 3, "s1": 1})

    assert researcher_runs(record) == {"s0": 3, "s1": 1}
    # The control: what the check used to read. A retry is invisible in it.
    assert {r.subtask_id: 1 for r in record.research_results} == {"s0": 1, "s1": 1}


@pytest.mark.parametrize(("runs", "passed"), [(2, True), (3, False)])
async def test_the_retry_cap_check_can_fail(
    judge_calls: list[str], runs: int, passed: bool
) -> None:
    """MAX_CRITIC_ITERATIONS=2 allows one first pass and one retry: 2 runs, not 3."""
    record = _record_with_researcher_runs({"s0": runs, "s1": 1})
    item = load_dataset().by_id(record.item_id)

    score = await score_run(item, record)
    check = next(c for c in score.tools.checks if c.name == "retry_count_within_cap")

    assert check.passed is passed


@pytest.mark.parametrize(("searches", "passed"), [(12, True), (13, False)])
async def test_the_search_ceiling_is_the_real_bound(
    judge_calls: list[str], searches: int, passed: bool
) -> None:
    """2 subtasks x 3 attempts x 2 runs each = 12. The old ceiling allowed 18."""
    item, plan = _item_and_plan(2)
    calls = [f"distinct search number {i} {'x' * i}" for i in range(searches)]

    score = await score_tools(item, plan, ["s0", "s1"], calls, {"s0": 2, "s1": 2})
    check = next(c for c in score.checks if c.name == "search_count_within_bounds")

    assert check.passed is passed


# --- the command's exit status --------------------------------------------------


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, crash_first: bool
) -> argparse.Namespace:
    async def fake_run_items(items: list[Any], concurrency: int = 5) -> list[RunRecord]:
        return [
            RunRecord(
                item_id=item.id,
                query=item.query,
                answer="" if crash_first and index == 0 else "an answer",
                error="ProviderError: x" if crash_first and index == 0 else None,
            )
            for index, item in enumerate(items)
        ]

    monkeypatch.setattr(cli, "run_items", fake_run_items)
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    return cli.build_parser().parse_args(["run", "--smoke", "--name", "t", "--write-baseline"])


def test_a_crashed_item_fails_the_run_command(
    judge_calls: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _patch_run(monkeypatch, tmp_path, crash_first=True)

    assert cli.cmd_run(args) == 1
    # The artefact is still written: it is what you debug the crash from.
    assert (tmp_path / "runs" / "t.json").exists() is True
    # A baseline with a crash baked in would make every later crash look normal.
    assert (tmp_path / "baseline.json").exists() is False


def test_a_clean_run_passes_and_writes_its_baseline(
    judge_calls: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The control: the same command, nothing crashed."""
    args = _patch_run(monkeypatch, tmp_path, crash_first=False)

    assert cli.cmd_run(args) == 0
    assert (tmp_path / "baseline.json").exists() is True
